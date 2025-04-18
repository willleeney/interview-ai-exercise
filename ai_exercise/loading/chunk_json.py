"""Basic function to chunk JSON data by key."""

from typing import Any


def chunk_data(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    info = data.get(key, {})
    return [{sub_key: sub_info} for sub_key, sub_info in info.items()]


def segmantic_chunk(json_data: dict[str: Any]) -> list[dict[str, Any]]:
    chunks = []
    
    # Handle paths - each endpoint becomes its own chunk
    for path, methods in json_data.get("paths", {}).items():
        for method, details in methods.items():
            endpoint_text = f"PATH: {path}\nMETHOD: {method}\n"
            endpoint_text += f"SUMMARY: {details.get('summary', '')}\n"
            endpoint_text += f"DESCRIPTION: {details.get('description', '')}\n"
            
            # Include parameters
            if "parameters" in details:
                endpoint_text += "PARAMETERS:\n"
                for param in details["parameters"]:
                    endpoint_text += f"- {param.get('name', '')}: {param.get('description', '')}\n"
            
            # Include request body if present
            if "requestBody" in details:
                endpoint_text += "REQUEST BODY:\n"
                endpoint_text += f"{str(details['requestBody'])[:200]}...\n"
            
            # Include responses
            endpoint_text += "RESPONSES:\n"
            for status, response in details.get("responses", {}).items():
                endpoint_text += f"- {status}: {response.get('description', '')}\n"
            
            chunks.append(endpoint_text)
    
    # Handle schemas - each schema becomes its own chunk
    for schema_name, schema in json_data.get("components", {}).get("schemas", {}).items():
        schema_text = f"SCHEMA: {schema_name}\n"
        schema_text += f"TYPE: {schema.get('type', '')}\n"
        
        # Include properties
        if "properties" in schema:
            schema_text += "PROPERTIES:\n"
            for prop_name, prop in schema["properties"].items():
                prop_type = prop.get("type", "object")
                prop_desc = prop.get("description", "")
                schema_text += f"- {prop_name} ({prop_type}): {prop_desc}\n"
        
        chunks.append(schema_text)
    
    return chunks