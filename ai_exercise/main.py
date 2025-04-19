"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import get_completion, create_prompt
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    bad_chunking,
    better_chunking
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
    EmptyDocumentsOutput
)
from ai_exercise.retrieval.vector_store import create_collection, empty_collection
from ai_exercise.retrieval.retrieval import get_relevant_chunks

app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/empty")
def empty_docs_route() -> HealthRouteOutput:
    """Route to empty the vector storage"""
    empty_collection(collection)
    assert collection.count() == 0
    return EmptyDocumentsOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store. """

    if SETTINGS.chunking_method == "bad":
        bad_chunking(collection) 
    elif SETTINGS.chunking_method == "better?":
        better_chunking(collection)

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
