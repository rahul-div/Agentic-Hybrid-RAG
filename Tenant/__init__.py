"""
Multi-Tenant RAG System

A production-ready, multi-tenant Retrieval-Augmented Generation (RAG) system
with complete data isolation using Neon PostgreSQL (project-per-tenant) and Neo4j + Graphiti.

Key Features:
- Complete tenant isolation using project-per-tenant architecture (Neon official best practice)
- Vector search with pgvector in isolated databases
- Knowledge graph capabilities with Graphiti namespace isolation
- JWT-based authentication and authorization
- FastAPI REST API with comprehensive endpoints
- Pydantic AI agent with tenant-aware tools
- Production-ready security and scalability patterns

Components:
- tenant_manager: Complete tenant lifecycle management with Neon project-per-tenant operations
- tenant_graphiti_client: Graph operations with group_id namespace isolation
- multi_tenant_agent: AI agent with tenant-aware tools
- multi_tenant_api: FastAPI application with authentication
- auth_middleware: JWT authentication and security middleware
"""

__version__ = "1.0.0"
__author__ = "Multi-Tenant RAG Team"
__license__ = "MIT"

# Core components
from .tenant_manager import (
    TenantManager,
    TenantCreateRequest,
    TenantInfo,
    TenantUsageStats,
    TenantError,
    TenantNotFoundError,
    TenantCreationError,
    TenantDeletionError,
    TenantStatus,
)

# Import Document and Chunk models (temporarily define here for compatibility)
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for tenant operations."""

    id: Optional[str] = None
    title: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Chunk(BaseModel):
    """Document chunk model."""

    id: Optional[str] = None
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: Optional[datetime] = None


from .tenant_graphiti_client import (
    TenantGraphitiClient,
    GraphEpisode,
    GraphEntity,
    GraphRelationship,
)
from .multi_tenant_agent import MultiTenantRAGAgent, TenantContext
from .auth_middleware import JWTManager, TenantAuthMiddleware, TenantSecurityManager

# API application
from .multi_tenant_api import create_app

# Main configuration
from .main import Config

__all__ = [
    # Core classes
    "TenantManager",
    "TenantCreateRequest",
    "TenantInfo",
    "TenantUsageStats",
    "TenantError",
    "TenantNotFoundError",
    "TenantCreationError",
    "TenantDeletionError",
    "TenantStatus",
    "Document",
    "Chunk",
    "TenantGraphitiClient",
    "GraphEpisode",
    "GraphEntity",
    "GraphRelationship",
    "MultiTenantRAGAgent",
    "TenantContext",
    "RAGResult",
    # Authentication
    "JWTManager",
    "TenantAuth",
    "TenantSecurityManager",
    # Application
    "create_app",
    "Config",
    # Version info
    "__version__",
    "__author__",
    "__license__",
]


def get_version():
    """Get the current version of the multi-tenant RAG system."""
    return __version__


def get_info():
    """Get system information."""
    return {
        "name": "Multi-Tenant RAG System",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Production-ready multi-tenant RAG with complete data isolation",
        "components": [
            "Neon PostgreSQL with pgvector (project-per-tenant)",
            "Neo4j with Graphiti (namespace isolation)",
            "FastAPI with JWT auth",
            "Pydantic AI agent",
            "Complete database isolation",
            "Namespace-based graph isolation",
        ],
    }
