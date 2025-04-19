
# from ragas.testset.transforms import NERExtractor
# from ragas.testset.transforms import apply_transforms, Parallel
# from ragas.testset.graph import Node, KnowledgeGraph
# from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder, CosineSimilarityBuilder
# from ragas.testset.transforms.relationship_builders.llm import SemanticTripleExtractor


"""
Main function to create synthetic database
"""

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.main import load_docs_route
from ai_exercise.retrieval.vector_store import create_collection
collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)

from ai_exercise.loading.document_loader import get_json_data
from ragas.testset.graph import Node, KnowledgeGraph, Relationship
from ragas.testset.transforms import JaccardSimilarityBuilder, KeyphrasesExtractor
from ragas.testset.transforms import apply_transforms, Parallel

from ragas.testset import TestsetGenerator

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o",
    client=openai_client 
))


from collections import defaultdict
import random

def extract_all_json_data() -> list[str]:
    """Extract json information from each api url"""
    api_docs = []
    for api_url in SETTINGS.docs_url:
        json_data = get_json_data(api_url)
        api_docs.append(json_data)
    return api_docs


def create_nodes_from_json(api_docs) -> list[Node]:
    """
    Create RAGAS Nodes from nested JSON data
    """
    nodes = []
    id_counter = 0
    
    def process_json_item(item, parent_id=None, path=""):
        nonlocal id_counter
        
        # Create a unique ID for this node
        node_id = f"node_{id_counter}"
        id_counter += 1
        
        # Initialize node properties
        properties = {}
        
        # Add basic metadata
        node_type = item.get("type", "api_connection")
        properties["type"] = node_type
        properties["id"] = node_id
        
        if parent_id is not None:
            properties["parent_id"] = parent_id
            properties["path"] = path
        
        # Process properties and nested items
        flat_content = {}
        nested_items = {}
        
        for key, value in item.items():
            if isinstance(value, dict):
                nested_items[key] = value
            elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
                nested_items[key] = value
            else:
                flat_content[key] = value
        
        # Create a readable page_content for the node
        page_content_parts = []
        for k, v in flat_content.items():
            if k != "type":  # Avoid duplication with the type property
                page_content_parts.append(f"{k}: {v}")
        
        properties["page_content"] = "\n".join(page_content_parts)
        properties["flat_content"] = flat_content
        
        # Create the node
        node = Node(properties=properties)
        nodes.append(node)
        
        # Process nested items
        for key, value in nested_items.items():
            if isinstance(value, dict):
                new_path = f"{path}.{key}" if path else key
                process_json_item(value, node_id, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        new_path = f"{path}.{key}[{i}]" if path else f"{key}[{i}]"
                        process_json_item(item, node_id, new_path)
    
    # Process each top-level item
    for item in api_docs:
        process_json_item(item)
    
    return nodes



def build_knowledge_graph(nodes: list[Node]) -> tuple[KnowledgeGraph, list[dict]]:
    """
    Build a RAGAS knowledge graph from nodes and apply relationships
    """
    # Create the knowledge graph
    kg = KnowledgeGraph(nodes=nodes)

    # Try to use entity similarity if entities were extracted
    entity_types = set()
    for node in nodes:
        if "entities" in node.properties:
            entity_types.update(node.properties["entities"].keys())
    
    transforms = []
    # Add Jaccard similarity for each entity type if available
    for entity_type in entity_types:
        entity_builder = JaccardSimilarityBuilder(
            property_name="entities",
            key_name=entity_type,
            new_property_name=f"{entity_type.lower()}_entity_jaccard_similarity",
            threshold=0.2
        )
        transforms.append(entity_builder)
    
    # Apply the transforms to the knowledge graph
    apply_transforms(kg, transforms)
    
    # Add parent-child relationships that were established during node creation
    parent_nodes = {node.id: node for node in kg.nodes}  # Create a lookup dictionary for nodes by ID

    for node in kg.nodes:
        if "parent_id" in node.properties:
            parent_id = node.properties["parent_id"]
            if parent_id in parent_nodes:  # Make sure the parent node exists
                parent_node = parent_nodes[parent_id]
                kg.add(Relationship(
                    source=parent_node,  # Use the actual parent node object
                    target=node,
                    type="contains",
                    properties={"confidence": 1.0}
                ))
        
    return kg

def generate_synthetic_queries(kg: KnowledgeGraph, num_queries: int = 10) -> list[str]:
    """
    Generate synthetic queries based only on the knowledge graph nodes
    """
    queries = []
    
    # Get all node types
    node_types = set(str(node.properties.get("type", "api_connection")) for node in kg.nodes)
    
    # Extract keyphrases directly from nodes if they exist
    all_keyphrases = set()
    for node in kg.nodes:
        if "keyphrases" in node.properties:
            all_keyphrases.update(node.properties["keyphrases"])
    
    # Get connection names or identifiers for more specific queries
    connection_names = []
    for node in kg.nodes:
        if "name" in node.properties:
            connection_names.append(node.properties["name"])
        elif "id" in node.properties and isinstance(node.properties["id"], str):
            connection_names.append(node.properties["id"])
    
    # Query templates based on node types
    templates = [
        "How do I configure the {entity_type} connection?",
        "What parameters are required for {entity_type}?",
        "What's the authentication method for {entity_type}?",
    ]
    
    # Add keyphrase templates if keyphrases were extracted
    if all_keyphrases:
        templates.extend([
            "How does {keyphrase} work in {entity_type}?",
            "Is {keyphrase} supported in {entity_type}?",
        ])
    
    # Add specific connection name templates if available
    if connection_names:
        templates.extend([
            "How do I connect to {connection_name}?",
            "What are the authentication options for {connection_name}?",
        ])
    
    # Generate queries using the available information
    count = 0
    
    while count < num_queries and count < len(templates) * len(node_types) * 2:
        template = random.choice(templates)
        
        if "{entity_type}" in template and "{keyphrase}" in template:
            if all_keyphrases and node_types:
                query = template.format(
                    entity_type=random.choice(list(node_types)),
                    keyphrase=random.choice(list(all_keyphrases))
                )
                queries.append(query)
                count += 1
        elif "{entity_type}" in template:
            if node_types:
                query = template.format(entity_type=random.choice(list(node_types)))
                queries.append(query)
                count += 1
        elif "{connection_name}" in template:
            if connection_names:
                query = template.format(connection_name=random.choice(connection_names))
                queries.append(query)
                count += 1
        elif "{keyphrase}" in template:
            if all_keyphrases:
                query = template.format(keyphrase=random.choice(list(all_keyphrases)))
                queries.append(query)
                count += 1
                
        # Stop if we've reached the requested number of queries
        if len(queries) >= num_queries:
            break
    
    return queries[:num_queries]

from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.llm.completions import get_completion, create_prompt

def main():
    query = "Can I filter employees by department when using the list endpoint?"

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

    # from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    # embedding_model = LlamaIndexEmbeddingsWrapper(openai_ef)

    # generator = TestsetGenerator(llm=evaluator_llm, embedding_model=embedding_model)
    # dataset = generator.generate(testset_size=10)


    api_docs = extract_all_json_data()

    nodes = create_nodes_from_json(api_docs)
    
    kg = build_knowledge_graph(nodes)

    queries = generate_synthetic_queries(kg, num_queries=20)

    print(queries)

    return


if __name__ == "__main__":
    main()
