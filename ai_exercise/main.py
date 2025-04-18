"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import get_completion, create_prompt
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data,
    split_docs
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
)
from ai_exercise.retrieval.vector_store import create_collection, empty_collection
from ai_exercise.retrieval.retrieval import get_relevant_chunks

app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to empty current collection and load documents into vector store. """
    empty_collection(collection)

    for api_url in SETTINGS.docs_url:
        # get the json data
        json_data = get_json_data(api_url)

        # build documents
        documents = build_docs(json_data)

        # split docs
        documents = split_docs(documents)

        # load documents into vector store
        spec_name = api_url.split("/")[-1].split(".")[0]
        add_documents(collection, documents, spec_name)
        print(f"Added {len(documents)} to collection")

        # check the number of documents in the collection
        print(f"Number of documents in collection: {collection.count()}")

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Get relevant chunks from the collection
    relevant_chunks = get_relevant_chunks(
        collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors
    )

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=relevant_chunks)
    print(f"Prompt: {prompt}")

    # Get completion from LLM
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )
    
    return ChatOutput(message=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
