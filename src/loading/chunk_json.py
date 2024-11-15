"""Basic function to chunk JSON data by key."""

from typing import Any


def chunk_data(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    info = data.get(key, {})
    return [{sub_key: sub_info} for sub_key, sub_info in info.items()]
