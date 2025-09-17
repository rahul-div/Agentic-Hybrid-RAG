"""
Interactive Multi-Tenant FastAPI for Hybrid RAG System
Provides authenticated endpoints for multi-tenant RAG with comprehensive agent integration.
"""

import logging
import os
import sys
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from uuid import UUID

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file from current directory
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("Environment variables will be read from system environment only")

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from jose import JWTError
    import uvicorn
except ImportError:
    print(
        "Warning: FastAPI dependencies not installed. Install with: pip install fastapi uvicorn[standard] python-jose[cryptography]"
    )
    sys.exit(1)

from tenant_manager import TenantManager
from auth_middleware import TenantContext, JWTAuthenticator
from multi_tenant_agent import MultiTenantRAGAgent

# Import validated agent from parent directory
try:
    pass  # Removed unused tool imports since we're using the agent directly
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Please ensure the agent folder is properly configured")
    sys.exit(1)

# Configure comprehensive logging like the reference comprehensive agent
# Create single console and file handlers to avoid duplication
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("interactive_multi_tenant_api.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Configure root logger with single handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()  # Clear any existing handlers
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Configure all backend loggers to inherit from root (avoiding handler duplication)
backend_loggers = [
    "multi_tenant_agent",
    "tenant_graphiti_client",
    "tenant_data_ingestion_service",
    "tenant_manager",
    "auth_middleware",
    "catalog_database",
    "google_genai",
    "httpx",
    "neo4j",
    "ingestion",
]

for logger_name in backend_loggers:
    backend_logger = logging.getLogger(logger_name)
    backend_logger.setLevel(logging.INFO)
    # Don't add handlers - let them inherit from root logger
    backend_logger.propagate = True

# Security scheme
security = HTTPBearer()

# Global state
tenant_manager = None
jwt_authenticator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global tenant_manager, jwt_authenticator

    logger.info("Starting Multi-Tenant RAG API...")

    try:
        # Initialize tenant manager with required parameters
        neon_api_key = os.getenv("NEON_API_KEY")
        catalog_db_url = os.getenv("CATALOG_DB_URL") or os.getenv("POSTGRES_URL")

        if not neon_api_key:
            logger.warning("NEON_API_KEY not found in environment variables")
        if not catalog_db_url:
            logger.warning(
                "CATALOG_DB_URL or POSTGRES_URL not found in environment variables"
            )

        # Initialize with environment variables or fallback values for testing
        neo4j_uri = (
            os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or "neo4j://127.0.0.1:7687"
        )
        neo4j_user = (
            os.getenv("NEO4J_USERNAME")
            or os.getenv("NEO4J_USER")
            or os.getenv("NEO4J_AUTH", "").split("/")[0]
            if os.getenv("NEO4J_AUTH")
            else "neo4j"
        )
        neo4j_password = os.getenv("NEO4J_PASSWORD") or (
            os.getenv("NEO4J_AUTH", "").split("/")[1]
            if "/" in os.getenv("NEO4J_AUTH", "")
            else None
        )

        # Debug Neo4j configuration
        logger.info(f"Neo4j URI: {neo4j_uri}")
        logger.info(f"Neo4j User: {neo4j_user}")
        logger.info(f"Neo4j Password: {'***' if neo4j_password else 'None'}")

        # Warn if credentials are missing
        if not neo4j_password:
            logger.warning(
                "⚠️  Neo4j password not found! Set NEO4J_PASSWORD environment variable for graph search."
            )
            logger.warning("   Example: export NEO4J_PASSWORD=your_neo4j_password")
            neo4j_uri = None  # Disable Neo4j if no password

        tenant_manager = TenantManager(
            neon_api_key=neon_api_key or "test_neon_api_key",
            catalog_db_url=catalog_db_url or "postgresql://localhost:5432/test_catalog",
            default_region="aws-us-east-1",
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        logger.info("Tenant manager initialized")

        # Initialize ingestion service for tenant-aware searches
        from tenant_data_ingestion_service import TenantDataIngestionService

        tenant_manager.ingestion_service = TenantDataIngestionService(
            tenant_manager=tenant_manager
        )
        logger.info("Tenant ingestion service initialized")

        # Initialize JWT authenticator
        jwt_authenticator = JWTAuthenticator()
        logger.info("JWT authenticator initialized")

        logger.info(
            "🎯 Multi-Tenant RAG API ready - all backend logs will appear in this terminal"
        )
        logger.info(
            "📊 Backend activity from CLI and API operations will be shown below:"
        )

        yield

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    finally:
        logger.info("Shutting down Multi-Tenant RAG API...")


# Initialize FastAPI app
app = FastAPI(
    title="Interactive Multi-Tenant RAG API",
    description="Authenticated FastAPI for multi-tenant hybrid RAG with knowledge graphs",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class AuthRequest(BaseModel):
    """Authentication request model."""

    tenant_id: str
    user_id: Optional[str] = None
    api_key: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response model."""

    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    expires_in: int


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    search_type: str = Field(
        default="comprehensive",
        description="Type of search: vector, graph, hybrid, comprehensive",
    )
    limit: int = Field(default=10, ge=1, le=50)
    text_weight: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Search result model."""

    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response model."""

    results: List[SearchResult]
    total_results: int
    search_type: str
    query: str
    tenant_id: str
    execution_time: float


class TenantCreateRequest(BaseModel):
    """Tenant creation request model."""

    name: str = Field(..., description="Human-readable tenant name")
    email: str = Field(..., description="Tenant email address")
    region: str = Field(default="aws-us-east-1", description="Neon region")
    plan: str = Field(default="basic", description="Tenant plan")


class TenantResponse(BaseModel):
    """Tenant response model."""

    tenant_id: str
    tenant_name: str
    tenant_email: str
    status: str
    created_at: datetime
    neon_project_id: Optional[str] = None
    region: str
    plan: str


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""

    document_id: str
    filename: str
    tenant_id: str
    status: str
    uploaded_at: datetime
    chunks_created: int
    processing_time_ms: float
    vector_stored: bool
    graph_stored: bool


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    session_id: Optional[str] = None
    search_preferences: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    session_id: str
    tenant_id: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    execution_time: float


class TenantInfo(BaseModel):
    """Tenant information model."""

    tenant_id: str
    status: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    tenant_count: int


# Dependency to get current tenant context
async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TenantContext:
    """Extract tenant context from JWT token."""
    try:
        # Decode JWT token
        payload = jwt_authenticator.decode_token(credentials.credentials)

        # Extract tenant context
        tenant_context = TenantContext(
            tenant_id=payload.get("tenant_id"),
            user_id=payload.get("user_id"),
            permissions=payload.get("permissions", ["read"]),
            metadata=payload.get("metadata", {}),
            session_id=payload.get("session_id"),
        )

        # Validate tenant exists
        if not await tenant_manager.tenant_exists(tenant_context.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_context.tenant_id} not found",
            )

        return tenant_context

    except JWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# API Endpoints


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Count active tenants
        tenant_count = len(await tenant_manager.list_tenants()) if tenant_manager else 0

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            tenant_count=tenant_count,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unhealthy",
        )


@app.post("/tenants", response_model=TenantResponse)
async def create_tenant(tenant_request: TenantCreateRequest):
    """Create a new tenant (public endpoint - no authentication required)."""
    try:
        from tenant_manager import TenantCreateRequest as TMTenantCreateRequest

        # Create tenant via TenantManager
        tm_request = TMTenantCreateRequest(
            name=tenant_request.name,
            email=tenant_request.email,
            region=tenant_request.region,
            plan=tenant_request.plan,
        )

        tenant_info = await tenant_manager.create_tenant(tm_request)

        return TenantResponse(
            tenant_id=str(tenant_info.tenant_id),
            tenant_name=tenant_info.tenant_name,
            tenant_email=tenant_info.tenant_email,
            status=tenant_info.status.value,
            created_at=tenant_info.created_at,
            neon_project_id=str(tenant_info.neon_project_id),
            region=tenant_info.region,
            plan=tenant_info.plan,
        )

    except Exception as e:
        logger.error(f"Failed to create tenant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant: {str(e)}",
        )


@app.post("/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest):
    """Authenticate and get access token."""
    try:
        # Validate tenant exists
        if not await tenant_manager.tenant_exists(auth_request.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {auth_request.tenant_id} not found",
            )

        # For demonstration, we'll use simple API key validation
        # In production, this would integrate with your auth provider
        expected_api_key = f"api_key_{auth_request.tenant_id}"
        if auth_request.api_key != expected_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )

        # Generate JWT token
        token_data = {
            "tenant_id": auth_request.tenant_id,
            "user_id": auth_request.user_id or "anonymous",
            "permissions": ["read", "write"],  # Default permissions
            "session_id": str(uuid.uuid4()),
            "exp": datetime.utcnow() + timedelta(hours=24),
        }

        access_token = jwt_authenticator.create_token(token_data)

        return AuthResponse(
            access_token=access_token,
            tenant_id=auth_request.tenant_id,
            expires_in=24 * 3600,  # 24 hours
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error",
        )


@app.get("/tenants/info", response_model=TenantInfo)
async def get_tenant_info(tenant_context: TenantContext = Depends(get_current_tenant)):
    """Get current tenant information."""
    try:
        tenant_data = await tenant_manager.get_tenant(tenant_context.tenant_id)

        if not tenant_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_context.tenant_id} not found",
            )

        # Convert from dataclass TenantInfo to Pydantic TenantInfo
        return TenantInfo(
            tenant_id=str(tenant_data.tenant_id),  # Convert UUID to string
            status=tenant_data.status.value
            if hasattr(tenant_data.status, "value")
            else str(tenant_data.status),
            created_at=tenant_data.created_at,
            metadata={
                "tenant_name": tenant_data.tenant_name,
                "tenant_email": tenant_data.tenant_email,
                "neon_project_id": tenant_data.neon_project_id,
                "region": tenant_data.region,
                "plan": tenant_data.plan,
                "max_documents": tenant_data.max_documents,
                "max_storage_mb": tenant_data.max_storage_mb,
                "updated_at": tenant_data.updated_at.isoformat()
                if tenant_data.updated_at
                else None,
            },
        )

    except Exception as e:
        logger.error(f"Error getting tenant info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant information",
        )


@app.post("/search", response_model=SearchResponse)
async def search(
    search_request: SearchRequest,
    tenant_context: TenantContext = Depends(get_current_tenant),
):
    """Perform multi-modal search based on type."""
    start_time = datetime.utcnow()

    try:
        # Use the multi-tenant agent for search operations to ensure proper tenant context
        agent = MultiTenantRAGAgent(
            tenant_manager=tenant_manager,
            graphiti_client=tenant_manager.graphiti_client,
        )

        # Convert string tenant_id to UUID for agent context
        from uuid import UUID

        tenant_uuid = (
            UUID(tenant_context.tenant_id)
            if isinstance(tenant_context.tenant_id, str)
            else tenant_context.tenant_id
        )

        # Import the dependencies class
        from multi_tenant_agent import TenantAgentDependencies

        # Create agent context with dependencies
        agent_deps = TenantAgentDependencies(
            tenant_id=str(tenant_uuid),
            session_id="search_session",
            user_id="api_user",
            tenant_manager=tenant_manager,
        )

        results = []

        if search_request.search_type == "vector":
            # Use direct vector search service call
            try:
                if hasattr(tenant_manager, "ingestion_service"):
                    tenant_db_url = await tenant_manager.get_tenant_database_url(
                        tenant_uuid
                    )
                    vector_results = (
                        await tenant_manager.ingestion_service.vector_search_for_tenant(
                            tenant_database_url=tenant_db_url,
                            query=search_request.query,
                            limit=search_request.limit,
                        )
                    )
                    results = [
                        SearchResult(
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            source=r.get("source", "vector_search"),
                            metadata=r.get("metadata", {}),
                        )
                        for r in vector_results
                    ]
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                results = []

        elif search_request.search_type == "graph":
            # Use direct graph search service call
            try:
                if tenant_manager.graphiti_client:
                    graph_results = await tenant_manager.graphiti_client.search(
                        tenant_id=str(tenant_uuid),
                        query=search_request.query,
                        limit=search_request.limit,
                    )
                    results = [
                        SearchResult(
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            source=r.get("source", "graph_search"),
                            metadata=r.get("metadata", {}),
                        )
                        for r in graph_results
                    ]
                else:
                    logger.warning("Graph search not available - no Graphiti client")
                    results = []
            except Exception as e:
                logger.error(f"Graph search failed: {e}")
                results = []

        elif search_request.search_type == "hybrid":
            # Use direct hybrid search (vector + keyword/BM25)
            try:
                if hasattr(tenant_manager, "ingestion_service"):
                    tenant_db_url = await tenant_manager.get_tenant_database_url(
                        tenant_uuid
                    )
                    hybrid_results = (
                        await tenant_manager.ingestion_service.hybrid_search_for_tenant(
                            tenant_database_url=tenant_db_url,
                            query=search_request.query,
                            limit=search_request.limit,
                            text_weight=search_request.text_weight,
                        )
                    )
                    results = [
                        SearchResult(
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            source=r.get("source", "hybrid_search"),
                            metadata=r.get("metadata", {}),
                        )
                        for r in hybrid_results
                    ]
                else:
                    logger.warning("Hybrid search not available - no ingestion service")
                    results = []
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                results = []
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                results = []

        elif search_request.search_type == "comprehensive":
            # Use comprehensive search (vector + graph + hybrid combined)
            try:
                all_results = []
                tenant_db_url = await tenant_manager.get_tenant_database_url(
                    tenant_uuid
                )

                # Vector search
                if hasattr(tenant_manager, "ingestion_service"):
                    try:
                        vector_results = await tenant_manager.ingestion_service.vector_search_for_tenant(
                            tenant_database_url=tenant_db_url,
                            query=search_request.query,
                            limit=search_request.limit
                            // 3,  # Split limit across methods
                        )
                        for r in vector_results:
                            all_results.append(
                                SearchResult(
                                    content=r.get("content", ""),
                                    score=r.get("score", 0.0),
                                    source="vector_search",
                                    metadata=r.get("metadata", {}),
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Vector search failed in comprehensive: {e}")

                # Graph search
                if tenant_manager.graphiti_client:
                    try:
                        graph_results = await tenant_manager.graphiti_client.search(
                            tenant_id=str(tenant_uuid),
                            query=search_request.query,
                            limit=search_request.limit // 3,
                        )
                        for r in graph_results:
                            all_results.append(
                                SearchResult(
                                    content=r.get("fact", r.get("content", "")),
                                    score=r.get("score", 0.0),
                                    source="graph_search",
                                    metadata=r.get("metadata", {}),
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Graph search failed in comprehensive: {e}")

                # Hybrid search
                if hasattr(tenant_manager, "ingestion_service"):
                    try:
                        hybrid_results = await tenant_manager.ingestion_service.hybrid_search_for_tenant(
                            tenant_database_url=tenant_db_url,
                            query=search_request.query,
                            limit=search_request.limit // 3,
                            text_weight=search_request.text_weight,
                        )
                        for r in hybrid_results:
                            all_results.append(
                                SearchResult(
                                    content=r.get("content", ""),
                                    score=r.get("score", 0.0),
                                    source="hybrid_search",
                                    metadata=r.get("metadata", {}),
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Hybrid search failed in comprehensive: {e}")

                # Sort all results by score and limit
                all_results.sort(key=lambda x: x.score, reverse=True)
                results = all_results[: search_request.limit]

            except Exception as e:
                logger.error(f"Comprehensive search failed: {e}")
                results = []

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid search type: {search_request.search_type}",
            )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type=search_request.search_type,
            query=search_request.query,
            tenant_id=tenant_context.tenant_id,
            execution_time=execution_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error for tenant {tenant_context.tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service error",
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    tenant_context: TenantContext = Depends(get_current_tenant),
):
    """Interactive chat with the RAG agent."""
    start_time = datetime.utcnow()

    try:
        # Initialize multi-tenant agent with tenant manager and graphiti client
        agent = MultiTenantRAGAgent(
            tenant_manager=tenant_manager,
            graphiti_client=tenant_manager.graphiti_client,
        )

        # Create tenant context for agent (convert string tenant_id to UUID)
        from uuid import UUID

        tenant_uuid = (
            UUID(tenant_context.tenant_id)
            if isinstance(tenant_context.tenant_id, str)
            else tenant_context.tenant_id
        )

        agent_context = TenantContext(
            tenant_id=tenant_uuid,  # Pass UUID, not string
            user_id=tenant_context.user_id,
            session_id=chat_request.session_id or str(uuid.uuid4()),
            metadata=chat_request.search_preferences or {},
        )

        # Get response from agent
        response = await agent.chat(message=chat_request.message, context=agent_context)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        return ChatResponse(
            response=response.get("response", ""),
            session_id=agent_context.session_id,
            tenant_id=str(
                tenant_context.tenant_id
            ),  # Convert back to string for response
            sources=response.get("sources", []),
            tools_used=response.get("tools_used", []),
            execution_time=execution_time,
        )

    except Exception as e:
        logger.error(f"Chat error for tenant {tenant_context.tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat service error",
        )


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingest_vector: bool = Form(True),
    ingest_graph: bool = Form(True),
    tenant_context: TenantContext = Depends(get_current_tenant),
):
    """Upload a document for the authenticated tenant with ingestion options."""
    try:
        # Validate that at least one ingestion option is selected
        if not ingest_vector and not ingest_graph:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one ingestion option (vector or graph) must be selected",
            )

        # Read file content
        content = await file.read()
        content_str = content.decode("utf-8", errors="ignore")

        # Create document upload request with ingestion options
        from tenant_ingestion_models import DocumentInput as ServiceDocumentRequest

        upload_request = ServiceDocumentRequest(
            title=file.filename or "Untitled Document",
            content=content_str,
            source="api_upload",
            metadata={
                "original_filename": file.filename,
                "content_type": file.content_type,
                "uploaded_at": datetime.utcnow().isoformat(),
            },
            ingest_vector=ingest_vector,
            ingest_graph=ingest_graph,
        )

        # Upload document using tenant ingestion service
        start_time = datetime.utcnow()

        # Convert tenant_id to string if needed
        tenant_id_str = (
            tenant_context.tenant_id
            if isinstance(tenant_context.tenant_id, str)
            else str(tenant_context.tenant_id)
        )

        # Log ingestion options
        ingestion_types = []
        if ingest_vector:
            ingestion_types.append("vector database")
        if ingest_graph:
            ingestion_types.append("knowledge graph")

        logger.info(
            f"Document upload for tenant {tenant_id_str}: {file.filename} -> {' + '.join(ingestion_types)}"
        )

        # Ingest document with specified options
        ingestion_result = (
            await tenant_manager.ingestion_service.ingest_document_for_tenant(
                tenant_id=tenant_id_str,
                document=upload_request,
            )
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DocumentUploadResponse(
            document_id=ingestion_result.document_id,
            filename=file.filename or "Untitled Document",
            tenant_id=tenant_id_str,
            status="uploaded",
            uploaded_at=datetime.utcnow(),
            chunks_created=ingestion_result.chunks_created,
            processing_time_ms=processing_time,
            vector_stored=ingestion_result.vector_stored,
            graph_stored=ingestion_result.graph_episode_created,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Document upload failed: {e}")

        # Check for specific API quota issues
        if (
            "quota" in error_msg.lower()
            or "rate limit" in error_msg.lower()
            or "429" in error_msg
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API quota exhausted. Please wait a few minutes or upgrade your Gemini API plan. "
                "Check https://ai.google.dev/gemini-api/docs/rate-limits for more details.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document upload failed: {str(e)}",
            )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Interactive Multi-Tenant RAG API",
        "version": "1.0.0",
        "description": "Authenticated FastAPI for multi-tenant hybrid RAG with knowledge graphs",
        "endpoints": {
            "health": "/health",
            "create_tenant": "POST /tenants",
            "auth": "/auth/login",
            "tenant_info": "/tenants/info",
            "search": "/search",
            "chat": "/chat",
            "upload_document": "POST /documents",
        },
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Multi-Tenant RAG API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting Interactive Multi-Tenant RAG API on {args.host}:{args.port}")

    uvicorn.run(
        "interactive_multi_tenant_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
