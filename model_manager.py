#!/usr/bin/env python3
"""
Model Manager for Ollama - Functions to list and switch between available models.
"""

import requests


class ModelManager:
    """Manages Ollama model operations."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def list_models(self) -> dict:
        """
        Get a list of available models from Ollama.
        
        Returns:
            Dictionary with 'models' list and 'error' if any.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("models", []):
                models.append({
                    "name": model.get("name", ""),
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", ""),
                    "digest": model.get("digest", "")[:12] if model.get("digest") else "",
                })
            
            return {"models": models, "error": None}
            
        except requests.exceptions.ConnectionError:
            return {
                "models": [],
                "error": "Could not connect to Ollama. Make sure it is running.",
            }
        except requests.exceptions.Timeout:
            return {"models": [], "error": "Request timed out."}
        except Exception as e:
            return {"models": [], "error": str(e)}

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the available models.
        
        Args:
            model_name: Name of the model to check.
            
        Returns:
            True if model exists, False otherwise.
        """
        result = self.list_models()
        if result["error"]:
            return False
        
        for model in result["models"]:
            # Match exact name or name without tag
            if model["name"] == model_name:
                return True
            # Handle case where user provides name without :latest tag
            if model["name"].split(":")[0] == model_name.split(":")[0]:
                return True
        
        return False

    def get_model_info(self, model_name: str) -> dict:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Dictionary with model info or error.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10,
            )
            response.raise_for_status()
            return {"info": response.json(), "error": None}
            
        except requests.exceptions.ConnectionError:
            return {
                "info": None,
                "error": "Could not connect to Ollama.",
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"info": None, "error": f"Model '{model_name}' not found."}
            return {"info": None, "error": str(e)}
        except Exception as e:
            return {"info": None, "error": str(e)}

    def format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

