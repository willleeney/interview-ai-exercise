"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.chunk_json import chunk_data, segmantic_chunk
from ai_exercise.models import Document


def get_json_data(api_url: str) -> dict[str, Any]:
    """Send a GET request to the URL specified by api_url"""
    response = requests.get(api_url)
    json_data = response.json()
    response.raise_for_status()

    return json_data


def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def chunks_to_documents(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an list of chunks into a list of Document objects."""
    return [
        Document(page_content=item, metadata={"source": source})
        for item in data
    ]

def add_documents(
    collection: chromadb.Collection, 
    docs: list[Document],
    spec_name: str) -> None:
    """Add documents to the collection"""
    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata or {} for doc in docs],
        ids=[f"{spec_name}_doc_{i}" for i in range(len(docs))],
    )


def bad_chunking(collection: chromadb.Collection):
    """Original chunking kept for comparison"""
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
    return

def better_chunking(collection: chromadb.Collection):
    """Chunking based on segmatic format of the json"""
    for api_url in SETTINGS.docs_url:
        # get the json data
        json_data = get_json_data(api_url)

        chunks = segmantic_chunk(json_data)

        # load documents into vector store
        spec_name = api_url.split("/")[-1].split(".")[0]
        documents = chunks_to_documents(chunks, spec_name)
        add_documents(collection, documents, spec_name)
    return