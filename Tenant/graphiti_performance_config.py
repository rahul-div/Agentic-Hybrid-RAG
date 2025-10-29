"""
Graphiti Performance Configuration
Centralized performance settings for optimized knowledge graph ingestion.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class GraphitiPerformanceConfig:
    """Performance configuration for Graphiti operations."""

    # LLM Configuration for deterministic, reliable processing
    llm_model: str = "gemini-2.5-flash"  # Stable model, no "thinking" variants
    temperature: float = 0.0  # Force deterministic output for KG reliability
    max_tokens: int = 2048  # Sufficient for structured JSON responses
    max_retries: int = 2  # Conservative retries for production
    timeout: int = 45  # Reasonable timeout for complex extraction

    # Content optimization settings
    max_content_length: int = 800  # Reduced for faster processing
    preserve_sentence_boundaries: bool = True  # Maintain readability

    # Batch processing settings
    batch_size: int = 3  # Process episodes in batches (reduced from 5)
    enable_batch_optimization: bool = True

    # Performance monitoring
    log_timing: bool = True
    log_performance_warnings: bool = True

    def to_llm_config(self) -> Dict[str, Any]:
        """Convert to LLM client configuration format."""
        return {
            "model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    def should_optimize_content(self, content: str) -> bool:
        """Check if content should be optimized for performance."""
        return len(content) > self.max_content_length

    def optimize_content(self, content: str) -> str:
        """Optimize content for faster processing while preserving meaning."""
        if not self.should_optimize_content(content):
            return content

        # Truncate at word boundary near the limit
        truncated = content[: self.max_content_length]

        if self.preserve_sentence_boundaries:
            # Try to end at a sentence boundary
            last_period = truncated.rfind(".")
            last_question = truncated.rfind("?")
            last_exclamation = truncated.rfind("!")

            last_sentence_end = max(last_period, last_question, last_exclamation)

            if (
                last_sentence_end > self.max_content_length * 0.7
            ):  # Better quality preservation
                truncated = truncated[: last_sentence_end + 1]

        return truncated

    @classmethod
    def get_optimized_config(cls) -> "GraphitiPerformanceConfig":
        """Get pre-configured production settings that prioritize reliability over speed."""
        return cls(
            llm_model="gemini-2.5-flash",  # Use stable model
            temperature=0.0,  # Force deterministic output
            max_tokens=2048,  # Sufficient for complete JSON responses
            max_retries=2,  # Conservative retries
            timeout=60,  # Longer timeout for reliability
            max_content_length=1500,  # Less aggressive truncation
            preserve_sentence_boundaries=True,
            batch_size=1,  # Process one episode at a time for atomicity
            enable_batch_optimization=False,  # Disable for consistency
            log_timing=True,
            log_performance_warnings=True,
        )

    @classmethod
    def get_quality_config(cls) -> "GraphitiPerformanceConfig":
        """Get configuration prioritizing extraction quality over speed."""
        return cls(
            llm_model="gemini-2.0-flash",  # Use available v1beta model for quality
            temperature=0.05,  # Very low temperature for consistency
            max_tokens=4096,  # Larger context for complex extractions
            max_retries=3,  # More retries for quality
            timeout=90,  # Longer timeout for complex processing
            max_content_length=2000,  # Less aggressive truncation
            preserve_sentence_boundaries=True,
            batch_size=1,  # Process one at a time for maximum quality
            enable_batch_optimization=False,
            log_timing=True,
            log_performance_warnings=True,
        )

    @classmethod
    def get_balanced_config(cls) -> "GraphitiPerformanceConfig":
        """Get balanced configuration between speed and quality."""
        return cls(
            llm_model="gemini-2.0-flash",
            temperature=0.0,
            max_tokens=2048,
            max_retries=2,
            timeout=60,
            max_content_length=1200,
            preserve_sentence_boundaries=True,
            batch_size=2,
            enable_batch_optimization=True,
            log_timing=True,
            log_performance_warnings=True,
        )


def get_optimized_graphiti_config() -> GraphitiPerformanceConfig:
    """
    Get the default optimized Graphiti configuration.

    Returns:
        GraphitiPerformanceConfig: Optimized configuration instance
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("🎯 Using BALANCED performance configuration for Graphiti")
    return GraphitiPerformanceConfig.get_balanced_config()


def apply_performance_optimizations() -> Dict[str, Any]:
    """
    Apply performance optimizations and return complete configuration.

    This function returns a dictionary with all necessary configurations
    for LLM, embedder, reranker, and performance settings.

    Returns:
        Dict containing all configuration sections
    """
    # Get base configuration
    config = get_optimized_graphiti_config()

    # Get model names from environment variables with production defaults
    llm_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "embedding-001")
    reranker_model = os.getenv(
        "GEMINI_RERANKER_MODEL", "gemini-2.5-flash"
    )  # Use same stable model

    return {
        "llm_config": {
            "model": llm_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_retries": config.max_retries,
            "timeout": config.timeout,
        },
        "embedder_config": {
            "model": embedding_model,  # Use environment variable for embedding model
        },
        "reranker_config": {
            "model": reranker_model,  # Use same stable model for consistency
            "temperature": 0.1,  # Lower temperature for consistency
            "max_tokens": 1536,  # Adequate for reranking
            "timeout": 30,
            "max_retries": 2,  # Consistent with LLM config
        },
        "performance_settings": {
            "max_content_length": config.max_content_length,
            "preserve_sentence_boundaries": config.preserve_sentence_boundaries,
            "batch_size": config.batch_size,
            "enable_batch_optimization": config.enable_batch_optimization,
        },
        "monitoring": {
            "log_timing": config.log_timing,
            "log_performance_warnings": config.log_performance_warnings,
        },
    }


def get_preset_config(preset: str = "optimized") -> Dict[str, Any]:
    """
    Get configuration for a specific preset.

    Args:
        preset: Configuration preset ('optimized', 'quality', 'balanced')

    Returns:
        Dict containing preset configuration
    """
    if preset == "optimized":
        config = GraphitiPerformanceConfig.get_optimized_config()
    elif preset == "quality":
        config = GraphitiPerformanceConfig.get_quality_config()
    elif preset == "balanced":
        config = GraphitiPerformanceConfig.get_balanced_config()
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Apply the same structure as apply_performance_optimizations
    llm_model = os.getenv("GEMINI_CHAT_MODEL", config.llm_model)
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "embedding-001")
    reranker_model = os.getenv("GEMINI_RERANKER_MODEL", "gemini-2.0-flash")

    return {
        "llm_config": {
            "model": llm_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_retries": config.max_retries,
            "timeout": config.timeout,
        },
        "embedder_config": {
            "model": embedding_model,
        },
        "reranker_config": {
            "model": reranker_model,
            "temperature": 0.1,
            "max_tokens": 1536,
            "timeout": 30,
            "max_retries": 2,
        },
        "performance_settings": {
            "max_content_length": config.max_content_length,
            "preserve_sentence_boundaries": config.preserve_sentence_boundaries,
            "batch_size": config.batch_size,
            "enable_batch_optimization": config.enable_batch_optimization,
        },
        "monitoring": {
            "log_timing": config.log_timing,
            "log_performance_warnings": config.log_performance_warnings,
        },
    }


# Default configuration instance
_default_config = None


def get_default_config() -> GraphitiPerformanceConfig:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = GraphitiPerformanceConfig.get_optimized_config()
    return _default_config
