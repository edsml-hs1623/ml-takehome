# I/O utilities
import json
from typing import Dict, Any, List
from pathlib import Path

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file
    """
    with open(Path(file_path), "r") as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file
    """
    with open(Path(file_path), "w") as f:
        json.dump(data, f)

def load_users(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load users from JSON file and return as dictionary keyed by user ID
    """
    users_data = load_json(file_path)
    return {user["id"]: user for user in users_data}

