"""
Flexible provider configuration for LLM and embedding models.
Uses Google Gemini for all AI tasks.
"""

import os
from typing import Optional
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> str:
    """
    Get LLM model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured Gemini model name
    """
    return model_choice or os.getenv("LLM_CHOICE", "gemini-2.0-flash")


def get_embedding_client() -> genai.Client:
    """
    Get embedding client configuration based on environment variables.

    Returns:
        Configured Gemini client for embeddings
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY must be set for Gemini embeddings"
        )

    return genai.Client(api_key=api_key)


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.

    Returns:
        Embedding model name (Gemini)
    """
    return os.getenv("EMBEDDING_MODEL", "embedding-001")


def get_ingestion_model() -> str:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).

    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = os.getenv("INGESTION_LLM_CHOICE")

    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()

    return get_llm_model(model_choice=ingestion_choice)


# Provider information functions
def get_llm_provider() -> str:
    """Get the LLM provider name."""
    return os.getenv("LLM_PROVIDER", "gemini")


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return os.getenv("EMBEDDING_PROVIDER", "gemini")


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        True if configuration is valid
    """
    required_vars = [
        "GOOGLE_API_KEY",  # or GEMINI_API_KEY
        "LLM_CHOICE",
        "EMBEDDING_MODEL",
    ]

    missing_vars = []

    # Check for Gemini API key (either form)
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing_vars.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")

    # Check other required vars
    for var in required_vars[1:]:  # Skip API key check since we handled it above
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False

    return True


def get_model_info() -> dict:
    """
    Get information about current model configuration.

    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": os.getenv("LLM_CHOICE", "gemini-2.0-flash"),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "ingestion_model": os.getenv("INGESTION_LLM_CHOICE", "same as main"),
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    }
