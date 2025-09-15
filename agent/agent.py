"""
Main Pydantic AI agent for agentic RAG with knowledge graph.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    onyx_search_tool,
    onyx_answer_with_quote_tool,
    comprehensive_search_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    OnyxSearchInput,
    OnyxAnswerInput,
    ComprehensiveSearchInput,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies for the hybrid RAG agent."""

    session_id: str
    onyx_service: Optional[Any] = None  # OnyxService instance for cloud search
    onyx_document_set_id: Optional[int] = None  # Document set ID for searches
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "use_onyx": True,
                "default_limit": 10,
                "onyx_retries": 7,  # Based on our validated configuration
            }


# Initialize the agent with flexible model configuration
rag_agent = Agent(
    get_llm_model(), deps_type=AgentDependencies, system_prompt=SYSTEM_PROMPT
)


# Register tools with proper docstrings (no description parameter)
@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies], query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for relevant information using semantic similarity.

    This tool performs vector similarity search across document chunks
    to find semantically related content. Returns the most relevant results
    regardless of similarity score.

    Args:
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)

    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    input_data = VectorSearchInput(query=query, limit=limit)

    results = await vector_search_tool(input_data)

    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]


@rag_agent.tool
async def graph_search(
    ctx: RunContext[AgentDependencies], query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for facts and relationships.

    This tool queries the knowledge graph to find specific facts, relationships
    between entities, and temporal information. Best for finding specific facts,
    relationships between companies/people/technologies, and time-based information.

    Args:
        query: Search query to find facts and relationships

    Returns:
        List of facts with associated episodes and temporal data
    """
    input_data = GraphSearchInput(query=query)

    results = await graph_search_tool(input_data)

    # Convert results to dict for agent
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "invalid_at": r.invalid_at,
            "source_node_uuid": r.source_node_uuid,
        }
        for r in results
    ]


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Perform both vector and keyword search for comprehensive results.

    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores. Best for combining semantic and exact matching.

    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)

    Returns:
        List of chunks ranked by combined relevance score
    """
    input_data = HybridSearchInput(query=query, limit=limit, text_weight=text_weight)

    results = await hybrid_search_tool(input_data)

    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies], document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.

    This tool fetches the full document content along with all its chunks
    and metadata. Best for getting comprehensive information from a specific
    source when you need the complete context.

    Args:
        document_id: UUID of the document to retrieve

    Returns:
        Complete document data with content and metadata, or None if not found
    """
    input_data = DocumentInput(document_id=document_id)

    document = await get_document_tool(input_data)

    if document:
        # Format for agent consumption
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"],
        }

    return None


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies], limit: int = 20, offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.

    This tool provides an overview of all documents in the knowledge base,
    including titles, sources, and chunk counts. Best for understanding
    what information sources are available.

    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination

    Returns:
        List of documents with metadata and chunk counts
    """
    input_data = DocumentListInput(limit=limit, offset=offset)

    documents = await list_documents_tool(input_data)

    # Convert to dict for agent
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in documents
    ]


@rag_agent.tool
async def get_entity_relationships(
    ctx: RunContext[AgentDependencies], entity_name: str, depth: int = 2
) -> Dict[str, Any]:
    """
    Get all relationships for a specific entity in the knowledge graph.

    This tool explores the knowledge graph to find how a specific entity
    (company, person, technology) relates to other entities. Best for
    understanding how companies or technologies relate to each other.

    Args:
        entity_name: Name of the entity to explore (e.g., "Google", "OpenAI")
        depth: Maximum traversal depth for relationships (1-5)

    Returns:
        Entity relationships and connected entities with relationship types
    """
    input_data = EntityRelationshipInput(entity_name=entity_name, depth=depth)

    return await get_entity_relationships_tool(input_data)


@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get the timeline of facts for a specific entity.

    This tool retrieves chronological information about an entity,
    showing how information has evolved over time. Best for understanding
    how information about an entity has developed or changed.

    Args:
        entity_name: Name of the entity (e.g., "Microsoft", "AI")
        start_date: Start date in ISO format (YYYY-MM-DD), optional
        end_date: End date in ISO format (YYYY-MM-DD), optional

    Returns:
        Chronological list of facts about the entity with timestamps
    """
    input_data = EntityTimelineInput(
        entity_name=entity_name, start_date=start_date, end_date=end_date
    )

    return await get_entity_timeline_tool(input_data)


@rag_agent.tool
async def onyx_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    num_results: int = 5,
    search_type: str = "hybrid",
) -> Dict[str, Any]:
    """
    Search for documents using Onyx Cloud's enterprise search capabilities.

    This tool provides access to Onyx's powerful document search with semantic
    understanding and relevance scoring. Use for finding specific documents
    or information across the indexed knowledge base.

    Args:
        query: Search query to find relevant documents
        num_results: Number of documents to return (1-10, default: 5)
        search_type: Type of search - 'hybrid', 'semantic', or 'keyword' (default: 'hybrid')

    Returns:
        Search results with documents, relevance scores, and metadata
    """
    input_data = OnyxSearchInput(
        query=query,
        num_results=min(max(num_results, 1), 10),  # Ensure valid range
        search_type=search_type,
    )

    # Use dependency injection for Onyx service and document set
    return await onyx_search_tool(
        input_data,
        onyx_service=ctx.deps.onyx_service,
        document_set_id=ctx.deps.onyx_document_set_id,
    )


@rag_agent.tool
async def onyx_answer_with_quote(
    ctx: RunContext[AgentDependencies],
    query: str,
    num_docs: int = 3,
    include_quotes: bool = True,
) -> Dict[str, Any]:
    """
    Get comprehensive answers with citations using Onyx Cloud's QA capabilities.

    This tool provides detailed answers with supporting quotes and source
    citations from the knowledge base. Best for questions that need
    authoritative answers backed by specific sources.

    Args:
        query: Question to answer using the knowledge base
        num_docs: Number of source documents to use (1-10, default: 3)
        include_quotes: Whether to include supporting quotes (default: True)

    Returns:
        Comprehensive answer with citations, quotes, and source documents
    """
    input_data = OnyxAnswerInput(
        query=query,
        num_docs=min(max(num_docs, 1), 10),  # Ensure valid range
        include_quotes=include_quotes,
    )

    # Use dependency injection for Onyx service and document set
    return await onyx_answer_with_quote_tool(
        input_data,
        onyx_service=ctx.deps.onyx_service,
        document_set_id=ctx.deps.onyx_document_set_id,
    )


@rag_agent.tool
async def comprehensive_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    num_results: int = 5,
    include_onyx: bool = True,
    include_vector: bool = True,
    include_graph: bool = True,
    search_type: str = "hybrid",
) -> Dict[str, Any]:
    """
    Perform comprehensive search across all available systems (Onyx, Vector, Knowledge Graph).

    This tool searches multiple systems simultaneously and combines results for
    the most comprehensive information retrieval. Use when you need to find
    information from all available sources or when a single search type might
    not provide complete coverage.

    Args:
        query: Search query to execute across all systems
        num_results: Number of results per search type (1-10, default: 5)
        include_onyx: Include Onyx cloud search (default: True)
        include_vector: Include vector similarity search (default: True)
        include_graph: Include knowledge graph search (default: True)
        search_type: Search type for compatible systems - 'hybrid', 'semantic', 'keyword'

    Returns:
        Combined results from all enabled search systems with unified summary
    """
    input_data = ComprehensiveSearchInput(
        query=query,
        num_results=min(max(num_results, 1), 10),  # Ensure valid range
        include_onyx=include_onyx,
        include_vector=include_vector,
        include_graph=include_graph,
        search_type=search_type,
    )

    # Use dependency injection for comprehensive search
    return await comprehensive_search_tool(
        input_data,
        onyx_service=ctx.deps.onyx_service,
        document_set_id=ctx.deps.onyx_document_set_id,
    )


# Helper function to initialize agent with Onyx integration
async def create_hybrid_agent_dependencies(
    session_id: str,
    cc_pair_id: int = 285,  # Use validated CC-pair
    user_id: Optional[str] = None,
    enable_onyx: bool = True,
) -> AgentDependencies:
    """
    Create AgentDependencies with properly initialized Onyx service.

    Based on our validated standalone implementation with document set caching
    and robust error handling. Uses proven CC-pair 285 by default.

    Args:
        session_id: Unique session identifier
        cc_pair_id: CC-pair ID for Onyx document set (default: 285 - validated)
        user_id: Optional user identifier
        enable_onyx: Whether to enable Onyx integration

    Returns:
        Configured AgentDependencies with Onyx service ready for hybrid RAG
    """
    dependencies = AgentDependencies(session_id=session_id, user_id=user_id)

    if enable_onyx:
        try:
            # Import and initialize validated Onyx service
            from onyx.service import OnyxService

            onyx_service = OnyxService()
            logger.info(
                f"🔄 Initializing Onyx Cloud integration for CC-pair {cc_pair_id}"
            )

            # Step 1: Test basic connectivity with a simple call
            try:
                # Test basic connectivity by checking if we can reach the API
                personas = onyx_service.get_personas()
                if not personas:
                    logger.error(f"❌ Onyx API not accessible - no personas returned")
                    dependencies.search_preferences["use_onyx"] = False
                    return dependencies
                else:
                    logger.info(
                        f"✅ Onyx API accessible - {len(personas)} personas found"
                    )
            except Exception as e:
                logger.error(f"❌ Onyx API connectivity test failed: {e}")
                dependencies.search_preferences["use_onyx"] = False
                return dependencies

            # Step 2: Verify CC-pair and get/create document set (validated method)
            try:
                # Use the working validated method from our standalone implementation
                document_set_id = onyx_service.create_document_set_validated(
                    cc_pair_id=cc_pair_id,
                    name=f"Hybrid_Agent_DocSet_{session_id}",
                    description=f"Document set for hybrid agent session {session_id}",
                )

                if document_set_id:
                    # Successfully initialized with existing/new document set
                    dependencies.onyx_service = onyx_service
                    dependencies.onyx_document_set_id = document_set_id
                    dependencies.search_preferences["use_onyx"] = True
                    dependencies.search_preferences["onyx_retries"] = (
                        3  # Reduced for hybrid system
                    )

                    logger.info(f"✅ Onyx integration initialized successfully")
                    logger.info(f"   • CC-pair: {cc_pair_id}")
                    logger.info(f"   • Document set: {document_set_id}")
                    logger.info(f"   • Status: Ready")
                else:
                    logger.warning(
                        f"❌ Failed to create/get document set for CC-pair {cc_pair_id}"
                    )
                    dependencies.search_preferences["use_onyx"] = False

            except Exception as e:
                logger.error(f"❌ Document set creation failed: {e}")
                dependencies.search_preferences["use_onyx"] = False

        except Exception as e:
            logger.error(f"❌ Failed to initialize Onyx service: {e}")
            dependencies.search_preferences["use_onyx"] = False
    else:
        dependencies.search_preferences["use_onyx"] = False
        logger.info("🔧 Onyx integration disabled by configuration")

    return dependencies


# Convenience function for quick agent setup
async def create_hybrid_rag_agent(
    session_id: str = None,
    cc_pair_id: int = 285,  # Use validated CC-pair by default
    enable_onyx: bool = True,
) -> tuple[Agent, AgentDependencies]:
    """
    Create a fully configured hybrid RAG agent with Onyx + Graphiti integration.

    Uses validated initialization patterns from our standalone Onyx agent to ensure
    robust setup with proper fallback to Graphiti-only mode if Onyx fails.

    Args:
        session_id: Session identifier (auto-generated if None)
        cc_pair_id: CC-pair for Onyx integration (default: 285 - validated)
        enable_onyx: Whether to enable Onyx integration

    Returns:
        Tuple of (Agent instance, configured dependencies)
    """
    if session_id is None:
        session_id = f"hybrid_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"🚀 Creating hybrid RAG agent for session: {session_id}")

    dependencies = await create_hybrid_agent_dependencies(
        session_id=session_id, cc_pair_id=cc_pair_id, enable_onyx=enable_onyx
    )

    # Log final configuration
    systems_enabled = []
    if dependencies.search_preferences.get("use_onyx"):
        systems_enabled.append("Onyx Cloud")
    if dependencies.search_preferences.get("use_vector"):
        systems_enabled.append("Vector DB")
    if dependencies.search_preferences.get("use_graph"):
        systems_enabled.append("Knowledge Graph")

    logger.info(f"🎯 Hybrid agent ready with: {', '.join(systems_enabled)}")

    return rag_agent, dependencies
