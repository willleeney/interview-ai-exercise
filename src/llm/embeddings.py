"""Embeddings using OpenAI"""

import chromadb.utils.embedding_functions as embedding_functions

from src.constants import SETTINGS

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=SETTINGS.openai_api_key.get_secret_value(),
    model_name=SETTINGS.embeddings_model,
)
