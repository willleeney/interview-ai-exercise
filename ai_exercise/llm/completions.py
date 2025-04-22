"""Generate a response using an LLM."""

from openai import OpenAI

def create_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context"""
    context_str = "\n\n".join(context)
    return f"""Please answer the question based on the following context:.

If question cannot be answered from context then gracefully acknowledge there is not enough information to answer:

Context:
{context_str}

Question: {query}

Answer:"""


def get_completion(client: OpenAI, prompt: str, model: str) -> str:
    """Get completion from OpenAI"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
