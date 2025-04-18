
"""
RAG Evaluation Script
This script sets up and runs the RAGAS evaluation
"""

from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.constants import SETTINGS, openai_client, chroma_client
from ai_exercise.retrieval.vector_store import create_collection
collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)

from ai_exercise.loading.document_loader import bad_chunking, better_chunking
from ai_exercise.retrieval.vector_store import empty_collection
from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.llm.completions import get_completion, create_prompt

import re
from tqdm import tqdm
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithoutReference
from langchain_openai.embeddings import OpenAIEmbeddings

def parse_question_list(text: str) -> list[str]:
    """
    Parse a numbered list of questions into a list of strings.
    """
    # Regular expression to match numbered questions
    # This pattern looks for:
    # - A number followed by a period and space at the beginning of a line
    # - Then captures all text until the next numbered question or end of string
    pattern = r'^\s*\d+\.\s*(.*?)(?=\n\s*\d+\.|$)'
    
    # Find all matches in the text using the regex pattern
    # re.MULTILINE makes ^ match the start of each line
    # re.DOTALL makes . match newlines as well
    questions = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    
    # Trim whitespace from each question
    questions = [q.strip() for q in questions]
    
    return questions


def generate_synth_testset(real_questions, n_synth=50):
    """Generate a list of synthetic questions from real questions. """
    query = (
        f"Given this list of example questions, generate {n_synth} more questions. "
        "The context these questions will be used in is to test a RAG system to help anwser question on an API spec."
        "Also given is some context on what the API spec looks like. \n"
    )

    example_questions = "Example questions: \n" + "\n".join(real_questions)

   
    relevant_chunks = get_relevant_chunks(
        collection=collection, query=query + example_questions, k=50
    )
    context = "Context: " + "\n\n".join(relevant_chunks)
    answer = "\nList of questions: "

    prompt = query + example_questions + context + answer
    
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    test_questions = parse_question_list(result)
    return test_questions


def generate_test_responses(test_questions: list) -> list[dict]:
    """Generate RAG system responses for each of the test questions"""

    dataset = []

    for query in tqdm(test_questions):
        # Get relevant chunks from the collection
        relevant_chunks = get_relevant_chunks(
            collection=collection, query=query, k=SETTINGS.k_neighbors
        )

        # Create prompt with context
        prompt = create_prompt(query=query, context=relevant_chunks)
    
        # Get completion from LLM
        result = get_completion(
            client=openai_client,
            prompt=prompt,
            model=SETTINGS.openai_model,
        )

        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_chunks,
                "response": result,
            }
        )

    return dataset


def run_evaluation(test_queries_dataset: list) -> dict:
    """Run the RAG evaluation on a set of test queries"""
    evaluation_dataset = EvaluationDataset.from_list(test_queries_dataset)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=SETTINGS.openai_model,
        client=openai_client,
        api_key=SETTINGS.openai_api_key.get_secret_value()
    ))

    result = evaluate(
        dataset=evaluation_dataset, 
        metrics=[LLMContextPrecisionWithoutReference(), 
                 ResponseRelevancy()],
        llm=evaluator_llm,
        embeddings=OpenAIEmbeddings(
            model=SETTINGS.embeddings_model, 
            api_key=SETTINGS.openai_api_key.get_secret_value())
    )
    
    print("Evaluation Metrics:")
    for metric, value in result._repr_dict.items():
        print(f"{metric}: {value:.4f}")
    
    return result


def main():
    """
    Main function to run the evaluation
    """
    real_questions = [
        "How do you authenticate to the StackOne API?",
        "Can I retrieve all linked accounts with workday provider?",
        "What is the default expiry of the session token?",
        "What fields must be sent to create a course on an LMS?",
        "What is the response body when listing an employee?",
    ]
    
    for chunk_fn in ["bad_chunking", "better_chunking"]:
        print(f"Testing chunking method: {chunk_fn}")
        print("Loading docs into storage")
        # Load docs into the storage
        empty_collection(collection)
        if chunk_fn in globals():
            globals()[chunk_fn](collection)
        else:
            print(f"Function {chunk_fn} not found")
            continue
        
        print("Generating test dataset")
        test_dataset = generate_synth_testset(real_questions)

        # Create example responses from test queries
        print("Generating RAG system reponses")
        test_dataset.extend(real_questions)
        dataset = generate_test_responses(test_dataset)

        # Run the evaluation
        print("Evaluating quality of responses")
        metrics = run_evaluation(dataset)
    

if __name__ == "__main__":
    main()