"""
Data models for multi-tenant ingestion service.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


class DocumentInput(BaseModel):
    """Input model for document ingestion."""

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source (file path, URL, etc.)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional document metadata"
    )
    # Ingestion options
    ingest_vector: bool = Field(
        default=True, description="Whether to ingest into vector database"
    )
    ingest_graph: bool = Field(
        default=True, description="Whether to ingest into knowledge graph"
    )


class DocumentChunk(BaseModel):
    """Model for document chunks."""

    content: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Number of tokens in chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class TenantIngestionResult(BaseModel):
    """Result of tenant document ingestion."""

    document_id: str = Field(..., description="Generated document ID")
    title: str = Field(..., description="Document title")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    vector_stored: bool = Field(
        default=True, description="Whether vector embeddings were stored successfully"
    )
    graph_stored: bool = Field(
        default=False, description="Whether graph data was stored successfully"
    )
    graph_episode_created: bool = Field(
        ..., description="Whether graph episode was created"
    )
    graph_episode_id: str | None = Field(
        default=None, description="Graph episode ID if created"
    )
    errors: list[str] = Field(
        default_factory=list, description="Any errors encountered"
    )


class BatchIngestionResult(BaseModel):
    """Result of batch document ingestion."""

    total_documents: int = Field(..., description="Total documents attempted")
    successful_documents: int = Field(
        ..., description="Successfully ingested documents"
    )
    failed_documents: int = Field(..., description="Failed document count")
    document_results: list[TenantIngestionResult] = Field(
        default_factory=list, description="Individual results"
    )
    total_processing_time_ms: float = Field(..., description="Total processing time")
    errors: list[str] = Field(default_factory=list, description="Batch-level errors")


@dataclass
class TenantAgentDependencies:
    """Dependencies injected per tenant for agent operations."""

    tenant_id: str
    tenant_database_url: str
    shared_graphiti_client: Any  # TenantGraphitiClient

    @classmethod
    async def create_for_tenant(
        cls, tenant_id: str, tenant_manager: Any, shared_graphiti_client: Any
    ) -> "TenantAgentDependencies":
        """
        Create tenant-specific dependencies.

        Args:
            tenant_id: UUID of the tenant
            tenant_manager: TenantManager instance
            shared_graphiti_client: Shared TenantGraphitiClient instance

        Returns:
            TenantAgentDependencies instance
        """
        # Get tenant-specific database URL
        db_url = await tenant_manager.get_tenant_database_url(tenant_id)

        return cls(
            tenant_id=str(tenant_id),
            tenant_database_url=db_url,
            shared_graphiti_client=shared_graphiti_client,
        )


class IngestionError(Exception):
    """Raised when document ingestion fails."""

    pass


class BatchIngestionError(Exception):
    """Raised when batch ingestion encounters errors."""

    def __init__(
        self,
        message: str,
        successful_ids: list[str] = None,
        failed_documents: list[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.successful_ids = successful_ids or []
        self.failed_documents = failed_documents or []
