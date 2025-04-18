
"""
RAG Evaluation Script
This script sets up and runs the RAGAS evaluation
"""

from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.constants import SETTINGS, openai_client, chroma_client
from ai_exercise.retrieval.vector_store import create_collection
collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)

from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.llm.completions import get_completion, create_prompt

from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o",
    client=openai_client 
))

from tqdm import tqdm 

def run_evaluation(test_queries_dataset: list):
    """
    Run the RAG evaluation on a set of test queries
    """
    # Run evaluation
    print("Starting evaluation...")
    evaluation_dataset = EvaluationDataset.from_list(test_queries_dataset)

    result = evaluate(
        dataset=evaluation_dataset, 
        metrics=[LLMContextPrecisionWithoutReference(), Faithfulness(), ResponseRelevancy()],
        llm=evaluator_llm
    )
    
    # Print results
    print("Evaluation completed")
    print("Aggregate Metrics:")
    for metric, value in result._repr_dict.items():
        print(f"{metric}: {value:.4f}")
    
    
    return result


def create_test_responses():
    test_queries = [
        "How do you authenticate to the StackOne API?",
        "Can I retrieve all linked accounts with workday provider?",
        "What is the default expiry of the session token?",
        "What fields must be sent to create a course on an LMS?",
        "What is the response body when listing an employee?",
    ]
    dataset = []

    for query in tqdm(test_queries):
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


def main():
    """
    Main function to run the evaluation
    """
    # Create example responses from test queries
    dataset = create_test_responses()

    # Run the evaluation
    metrics = run_evaluation(dataset)
    

if __name__ == "__main__":
    main()