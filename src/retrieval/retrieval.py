"""Retrieve relevant chunks from a vector store."""

import chromadb


def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list[str]:
    """Retrieve k most relevant chunks for the query"""
    results = collection.query(query_texts=[query], n_results=k)

    return results["documents"][0]
