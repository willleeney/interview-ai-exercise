"""Create a vector store."""

from typing import Any

import chromadb


def create_collection(
    client: chromadb.Client, embedding_fn: Any, name: str
) -> chromadb.Collection:
    """Create and return a Chroma collection, or get existing one if it exists"""
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn)
