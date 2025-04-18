"""Create a vector store."""

from typing import Any

import chromadb


def create_collection(
    client: chromadb.Client, embedding_fn: Any, name: str
) -> chromadb.Collection:
    """Create and return a Chroma collection, or get existing one if it exists"""
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn)


def empty_collection(collection: chromadb.Collection):
    """Empties a Chroma collection if it is not empty"""
    all_ids = collection.get()["ids"]

    # Check if there are any documents to delete
    if all_ids:
        collection.delete(ids=all_ids)
        print(f"Successfully deleted {len(all_ids)} documents from the collection.")
    else:
        print("Collection is already empty, nothing to delete.")

    return