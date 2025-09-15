"""
Tools for the Pydantic AI agent.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks,
)# Your dual entity extraction strategy
# 1. Regex extraction (structured data)
entities = {
    "clients": ["TechCorp Inc"],
    "projects": ["MOBILE-APP-123"], 
    "team_members": ["John Smith"],
    "technologies": ["React Native", "PostgreSQL"]
}

# 2. Flattened for Graphiti context
flattened_metadata = {
    "entities_clients": "TechCorp Inc",
    "entities_projects": "MOBILE-APP-123",
    "entities_team_members": "John Smith",
    "entities_technologies": "React Native, PostgreSQL"
}

# 3. Graphiti's LLM enhancement
# Creates rich semantic understanding:
# - John Smith WORKS_ON MOBILE-APP-123 (relationship)
# - MOBILE-APP-123 HAS_CLIENT TechCorp Inc (business relationship)
# - MOBILE-APP-123 USES_TECHNOLOGY React Native (technical relationship)
# - All with temporal validity and confidence scores
from .graph_utils import search_knowledge_graph, get_entity_relationships, graph_client
from .models import ChunkResult, GraphSearchResult, DocumentMetadata

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Gemini via Graphiti.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    try:
        # Import here to avoid circular import issues
        from ingestion.embedder import gemini_embedder

        # Use Graphiti's GeminiEmbedder directly
        embedding = await gemini_embedder.create(text)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class GraphSearchInput(BaseModel):
    """Input for graph search tool."""

    query: str = Field(..., description="Search query")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(
        default=0.3, description="Weight for text similarity (0-1)"
    )


class DocumentInput(BaseModel):
    """Input for document retrieval."""

    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""

    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


class EntityRelationshipInput(BaseModel):
    """Input for entity relationship query."""

    entity_name: str = Field(..., description="Name of the entity")
    depth: int = Field(default=2, description="Maximum traversal depth")


class EntityTimelineInput(BaseModel):
    """Input for entity timeline query."""

    entity_name: str = Field(..., description="Name of the entity")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


class OnyxSearchInput(BaseModel):
    """Input for Onyx search tool."""

    query: str = Field(..., description="Search query")
    num_results: int = Field(default=5, description="Number of results to return (1-10)")
    search_type: str = Field(default="hybrid", description="Type of search: 'hybrid', 'semantic', or 'keyword'")


class OnyxAnswerInput(BaseModel):
    """Input for Onyx answer with quote tool."""

    query: str = Field(..., description="Question to answer")
    num_docs: int = Field(default=5, description="Number of documents to use for answering")
    include_quotes: bool = Field(default=True, description="Whether to include supporting quotes")
    search_type: str = Field(default="hybrid", description="Type of search: 'hybrid', 'semantic', or 'keyword'")

class ComprehensiveSearchInput(BaseModel):
    """Input for comprehensive search combining multiple systems."""

    query: str = Field(..., description="Search query")
    num_results: int = Field(default=5, description="Number of results per search type")
    include_onyx: bool = Field(default=True, description="Include Onyx cloud search")
    include_vector: bool = Field(default=True, description="Include vector search") 
    include_graph: bool = Field(default=True, description="Include knowledge graph search")
    search_type: str = Field(default="hybrid", description="Search type for compatible systems")


class LocalDualSearchInput(BaseModel):
    """Input for Local Path: Dual Storage search (vector + graph only)."""
    
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of vector results")
    use_vector: bool = Field(default=True, description="Include vector similarity search")
    use_graph: bool = Field(default=True, description="Include knowledge graph search")


# Tool Implementation Functions
async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Perform vector similarity search.

    Args:
        input_data: Search parameters

    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)

        # Perform vector search
        results = await vector_search(embedding=embedding, limit=input_data.limit)

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def graph_search_tool(input_data: GraphSearchInput) -> List[GraphSearchResult]:
    """
    Search the knowledge graph.

    Args:
        input_data: Search parameters

    Returns:
        List of graph search results
    """
    try:
        results = await search_knowledge_graph(query=input_data.query)

        # Convert to GraphSearchResult models
        return [
            GraphSearchResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid"),
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Perform hybrid search (vector + keyword).

    Args:
        input_data: Search parameters

    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)

        # Perform hybrid search
        results = await hybrid_search(
            embedding=embedding,
            query_text=input_data.query,
            limit=input_data.limit,
            text_weight=input_data.text_weight,
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document.

    Args:
        input_data: Document retrieval parameters

    Returns:
        Document data or None
    """
    try:
        document = await get_document(input_data.document_id)

        if document:
            # Also get all chunks for the document
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks

        return document

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    List available documents.

    Args:
        input_data: Listing parameters

    Returns:
        List of document metadata
    """
    try:
        documents = await list_documents(
            limit=input_data.limit, offset=input_data.offset
        )

        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count"),
            )
            for d in documents
        ]

    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []


async def get_entity_relationships_tool(
    input_data: EntityRelationshipInput,
) -> Dict[str, Any]:
    """
    Get relationships for an entity.

    Args:
        input_data: Entity relationship parameters

    Returns:
        Entity relationships
    """
    try:
        return await get_entity_relationships(
            entity=input_data.entity_name, depth=input_data.depth
        )

    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        return {
            "central_entity": input_data.entity_name,
            "related_entities": [],
            "relationships": [],
            "depth": input_data.depth,
            "error": str(e),
        }


async def get_entity_timeline_tool(
    input_data: EntityTimelineInput,
) -> List[Dict[str, Any]]:
    """
    Get timeline of facts for an entity.

    Args:
        input_data: Timeline query parameters

    Returns:
        Timeline of facts
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None

        if input_data.start_date:
            start_date = datetime.fromisoformat(input_data.start_date)
        if input_data.end_date:
            end_date = datetime.fromisoformat(input_data.end_date)
        # Get timeline from graph
        timeline = await graph_client.get_entity_timeline(
            entity_name=input_data.entity_name, start_date=start_date, end_date=end_date
        )

        return timeline

    except Exception as e:
        logger.error(f"Entity timeline query failed: {e}")
        return []


# Enhanced Local Path: Dual Storage Synthesis Tool
async def local_dual_search_tool(input_data: LocalDualSearchInput) -> Dict[str, Any]:
    """
    SMART Local Path: Dual Storage search combining vector similarity + knowledge graph relationships.
    
    This tool performs intelligent synthesis of:
    - 🔍 pgvector semantic similarity search (contextual content)  
    - 🕸️ Neo4j + Graphiti knowledge graph search (hidden relationships & insights)
    
    Returns a synthesized response that combines semantic similarity with relationship discovery,
    complete with source citations and confidence scoring.
    
    Args:
        input_data: Search parameters for local dual storage search
        
    Returns:
        Dictionary with synthesized answer, source citations, and relationship insights
    """
    try:
        results = {
            "search_type": "local_dual_storage_synthesis",
            "query": input_data.query,
            "synthesized_answer": "",
            "vector_results": [],
            "graph_results": [], 
            "source_citations": [],
            "relationship_insights": [],
            "confidence_score": 0.0,
            "synthesis_metadata": {
                "vector_sources": 0,
                "graph_facts": 0,
                "synthesis_quality": "unknown"
            }
        }
        
        # Execute both searches in parallel for optimal performance
        tasks = []
        
        if input_data.use_vector:
            tasks.append(("vector", vector_search_tool(VectorSearchInput(
                query=input_data.query, 
                limit=input_data.limit
            ))))
            
        if input_data.use_graph:
            tasks.append(("graph", graph_search_tool(GraphSearchInput(
                query=input_data.query
            ))))
        
        if not tasks:
            results["synthesized_answer"] = "No search systems enabled."
            return results
            
        # Parallel execution
        search_results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        # Process results
        vector_data = []
        graph_data = []
        
        for i, (search_type, search_result) in enumerate(zip([t[0] for t in tasks], search_results)):
            if isinstance(search_result, Exception):
                logger.error(f"❌ {search_type} search failed: {search_result}")
                continue
                
            if search_type == "vector" and search_result:
                vector_data = search_result
                results["vector_results"] = [
                    {
                        "content": chunk.content,
                        "score": chunk.score,
                        "source": chunk.document_title,
                        "document_id": chunk.document_id
                    }
                    for chunk in search_result[:5]  # Top 5 results
                ]
                results["synthesis_metadata"]["vector_sources"] = len(search_result)
                
            elif search_type == "graph" and search_result:
                graph_data = search_result
                results["graph_results"] = [
                    {
                        "fact": fact.fact,
                        "confidence": getattr(fact, 'confidence', 'medium'),
                        "source": getattr(fact, 'source_node_uuid', 'knowledge_graph')
                    }
                    for fact in search_result[:5]  # Top 5 facts
                ]
                results["synthesis_metadata"]["graph_facts"] = len(search_result)
        
        # INTELLIGENT SYNTHESIS: Combine semantic similarity with relationship insights
        synthesis_parts = []
        source_citations = []
        confidence_scores = []
        
        # 1. Primary content from highest-scoring vector results
        if vector_data:
            best_chunks = sorted(vector_data, key=lambda x: x.score, reverse=True)[:3]
            
            primary_content = []
            for i, chunk in enumerate(best_chunks):
                if chunk.score > 0.7:  # High-confidence threshold
                    content_preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
                    primary_content.append(content_preview)
                    source_citations.append({
                        "type": "semantic_similarity",
                        "source": chunk.document_title,
                        "relevance_score": chunk.score,
                        "document_id": chunk.document_id
                    })
                    confidence_scores.append(chunk.score)
            
            if primary_content:
                synthesis_parts.append(f"**Primary Content (Semantic Search):** {' '.join(primary_content[:2])}")
        
        # 2. Relationship insights from knowledge graph
        if graph_data:
            relationship_insights = []
            for fact in graph_data[:3]:  # Top 3 relationship facts
                if hasattr(fact, 'fact') and fact.fact:
                    relationship_insights.append(fact.fact)
                    source_citations.append({
                        "type": "knowledge_graph_relationship", 
                        "source": "Knowledge Graph",
                        "fact": fact.fact,
                        "uuid": getattr(fact, 'uuid', 'unknown')
                    })
            
            if relationship_insights:
                results["relationship_insights"] = relationship_insights
                relationships_text = " | ".join(relationship_insights[:2])
                synthesis_parts.append(f"**Hidden Relationships & Insights:** {relationships_text}")
        
        # 3. Create unified intelligent synthesis
        if synthesis_parts:
            results["synthesized_answer"] = "\n\n".join(synthesis_parts)
            
            # Calculate confidence based on multi-system validation
            if vector_data and graph_data:
                results["synthesis_metadata"]["synthesis_quality"] = "high_multi_system"
                avg_vector_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
                graph_boost = 0.2 if graph_data else 0  # Knowledge graph adds confidence
                results["confidence_score"] = min(avg_vector_score + graph_boost, 1.0)
                
            elif vector_data:
                results["synthesis_metadata"]["synthesis_quality"] = "medium_vector_only"
                results["confidence_score"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
                
            elif graph_data:
                results["synthesis_metadata"]["synthesis_quality"] = "medium_graph_only"
                results["confidence_score"] = 0.6  # Knowledge graph provides good insights
                
        else:
            results["synthesized_answer"] = f"No relevant information found for query: {input_data.query}"
            results["synthesis_metadata"]["synthesis_quality"] = "no_results"
            results["confidence_score"] = 0.0
        
        # 4. Finalize source citations
        results["source_citations"] = source_citations
        
        logger.info(f"🎯 Local Dual Storage synthesis: {results['synthesis_metadata']['vector_sources']} vector + {results['synthesis_metadata']['graph_facts']} graph sources")
        
        return results
        
    except Exception as e:
        logger.error(f"Local dual search synthesis failed: {e}")
        return {
            "search_type": "local_dual_storage_synthesis", 
            "query": input_data.query,
            "synthesized_answer": f"Search synthesis failed due to error: {str(e)}",
            "vector_results": [],
            "graph_results": [],
            "source_citations": [],
            "relationship_insights": [],
            "confidence_score": 0.0,
            "error": str(e)
        }


# Combined search function for agent use
async def perform_comprehensive_search(
    query: str, use_vector: bool = True, use_graph: bool = True, limit: int = 10
) -> Dict[str, Any]:
    """
    Perform a comprehensive search using multiple methods.

    Args:
        query: Search query
        use_vector: Whether to use vector search
        use_graph: Whether to use graph search
        limit: Maximum results per search type (only applies to vector search)

    Returns:
        Combined search results
    """
    results = {
        "query": query,
        "vector_results": [],
        "graph_results": [],
        "total_results": 0,
    }

    tasks = []

    if use_vector:
        tasks.append(vector_search_tool(VectorSearchInput(query=query, limit=limit)))

    if use_graph:
        tasks.append(graph_search_tool(GraphSearchInput(query=query)))

    if tasks:
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        if use_vector and not isinstance(search_results[0], Exception):
            results["vector_results"] = search_results[0]

        if use_graph:
            graph_idx = 1 if use_vector else 0
            if not isinstance(search_results[graph_idx], Exception):
                results["graph_results"] = search_results[graph_idx]

    results["total_results"] = len(results["vector_results"]) + len(
        results["graph_results"]
    )

    return results


# Onyx Cloud Integration Tools - VALIDATED IMPLEMENTATION  
# Based on successful standalone interactive_onyx_agent.py implementation
# Uses proven document set strategy with robust fallback to Graphiti/pgvector

async def onyx_search_tool(input_data: OnyxSearchInput, onyx_service=None, document_set_id=None) -> Dict[str, Any]:
    """
    Search documents using Onyx Cloud's validated enterprise search capabilities.
    
    Uses the proven search_with_document_set_validated method from our standalone implementation
    with intelligent fallback to simple_chat when document set search fails.

    Args:
        input_data: Search parameters
        onyx_service: Injected OnyxService instance (dependency injection)
        document_set_id: Document set ID for targeted search

    Returns:
        Dictionary containing search results with documents and metadata
    """
    try:
        # Require dependency injection - no fallback service creation
        if onyx_service is None:
            logger.warning("No OnyxService provided via dependency injection")
            return {
                "search_type": "onyx_search_failed",
                "query": input_data.query,
                "answer": "",
                "source_documents": [],
                "total_found": 0,
                "success": False,
                "error": "OnyxService not available - dependency injection required",
                "fallback_recommended": True
            }
        
        # Primary: Use validated document set search if available
        if document_set_id:
            logger.info(f"Using Onyx document set {document_set_id} for query: {input_data.query}")
            search_results = await asyncio.to_thread(
                onyx_service.search_with_document_set_validated,
                query=input_data.query,
                document_set_id=document_set_id,
                max_retries=3  # Reduced retries for hybrid system responsiveness
            )
            
            if search_results.get("success"):
                source_docs = search_results.get("source_documents", [])
                return {
                    "search_type": "onyx_search_document_set",
                    "query": input_data.query,
                    "answer": search_results.get("answer", ""),
                    "source_documents": source_docs,
                    "document_set_id": document_set_id,
                    "attempt": search_results.get("attempt", 1),
                    "total_found": len(source_docs),
                    "success": True,
                    "confidence": "high"  # Document set search is most reliable
                }
            else:
                logger.warning(f"Document set search failed: {search_results.get('error', 'Unknown error')}")
        
        # Secondary fallback: Use simple_chat for general search
        logger.info(f"Falling back to Onyx simple_chat for query: {input_data.query}")
        simple_result = await asyncio.to_thread(
            onyx_service.simple_chat,
            message=f"Search for information about: {input_data.query}"
        )
        
        if simple_result and simple_result.strip():
            return {
                "search_type": "onyx_search_simple",
                "query": input_data.query,
                "answer": simple_result,
                "source_documents": [],
                "total_found": 1,
                "success": True,
                "confidence": "medium",
                "fallback_used": True
            }
        else:
            # Complete fallback failure - recommend hybrid search
            return {
                "search_type": "onyx_search_failed",
                "query": input_data.query,
                "answer": "",
                "source_documents": [],
                "total_found": 0,
                "success": False,
                "error": "Both document set and simple search failed",
                "fallback_recommended": True,
                "confidence": "none"
            }
        
    except Exception as e:
        logger.error(f"Onyx search failed with exception: {e}")
        return {
            "search_type": "onyx_search_error",
            "query": input_data.query,
            "answer": "",
            "source_documents": [],
            "total_found": 0,
            "success": False,
            "error": str(e),
            "fallback_recommended": True
        }


async def onyx_answer_with_quote_tool(input_data: OnyxAnswerInput, onyx_service=None, document_set_id=None) -> Dict[str, Any]:
    """
    Get answers with supporting quotes using Onyx Cloud's validated QA capabilities.
    
    Uses the proven answer_with_quote method from our standalone implementation
    with intelligent fallback patterns for hybrid system integration.

    Args:
        input_data: Question and answering parameters
        onyx_service: Injected OnyxService instance (dependency injection)
        document_set_id: Document set ID for targeted search

    Returns:
        Dictionary containing answer, quotes, and source documents
    """
    try:
        # Require dependency injection - no fallback service creation
        if onyx_service is None:
            logger.warning("No OnyxService provided via dependency injection") 
            return {
                "search_type": "onyx_answer_failed",
                "query": input_data.query,
                "answer": "",
                "answer_citationless": "",
                "quotes": [],
                "source_documents": [],
                "success": False,
                "error": "OnyxService not available - dependency injection required",
                "fallback_recommended": True
            }
        
        # Primary: Try document set search if available
        if document_set_id:
            logger.info(f"Using Onyx document set {document_set_id} for question: {input_data.query}")
            search_results = await asyncio.to_thread(
                onyx_service.search_with_document_set_validated,
                query=input_data.query,
                document_set_id=document_set_id,
                max_retries=3
            )
            
            if search_results.get("success"):
                source_docs = search_results.get("source_documents", [])
                answer = search_results.get("answer", "")
                return {
                    "search_type": "onyx_answer_document_set", 
                    "query": input_data.query,
                    "answer": answer,
                    "answer_citationless": answer,  # Document set doesn't separate citations
                    "source_documents": source_docs,
                    "document_set_id": document_set_id,
                    "attempt": search_results.get("attempt", 1),
                    "success": True,
                    "confidence": "high",
                    "quotes": []  # Document set format doesn't provide quote structure
                }
            else:
                logger.warning(f"Document set answer failed: {search_results.get('error', 'Unknown error')}")
        
        # Secondary: Use answer_with_quote method for comprehensive response  
        logger.info(f"Using Onyx answer_with_quote for question: {input_data.query}")
        answer_response = await asyncio.to_thread(
            onyx_service.answer_with_quote,
            query=input_data.query,
            num_docs=input_data.num_docs,
            include_quotes=input_data.include_quotes
        )
        
        if answer_response and answer_response.get('answer'):
            return {
                "search_type": "onyx_answer_comprehensive",
                "query": input_data.query,
                "answer": answer_response.get('answer', ''),
                "answer_citationless": answer_response.get('answer_citationless', ''),
                "quotes": answer_response.get('quotes', []),
                "source_documents": answer_response.get('top_documents', []),
                "contexts": answer_response.get('contexts', {}),
                "success": True,
                "confidence": "medium",
                "search_parameters": {
                    "num_docs": input_data.num_docs,
                    "include_quotes": input_data.include_quotes,
                    "search_type": input_data.search_type
                }
            }
        else:
            # Complete failure - recommend hybrid fallback
            return {
                "search_type": "onyx_answer_failed",
                "query": input_data.query,
                "answer": "",
                "answer_citationless": "",
                "quotes": [],
                "source_documents": [],
                "success": False,
                "error": "Both document set and answer_with_quote failed",
                "fallback_recommended": True,
                "confidence": "none"
            }
    
    except Exception as e:
        logger.error(f"Onyx answer with quote failed with exception: {e}")
        return {
            "search_type": "onyx_answer_error",
            "query": input_data.query,
            "answer": f"Unable to answer question due to service error: {str(e)}",
            "answer_citationless": f"Service error occurred",
            "quotes": [],
            "source_documents": [],
            "contexts": {},
            "success": False,
            "error": str(e),
            "fallback_recommended": True
        }


async def comprehensive_search_tool(input_data: ComprehensiveSearchInput, onyx_service=None, document_set_id=None) -> Dict[str, Any]:
    """
    Perform comprehensive search combining Local Path: Dual Storage and Onyx systems.
    
    Implements LOCAL-FIRST strategy based on system configuration:
    1. Try Local Path: Dual Storage first (pgvector + Neo4j + Graphiti)  
    2. Use Onyx Cloud as enhancement/fallback for enterprise coverage
    3. Combine results intelligently with local results taking priority
    
    Note: Current implementation still executes Onyx first for comprehensive_search tool,
    but system prompt directs agents to prioritize individual local tools (vector_search, graph_search) over comprehensive_search for most queries.

    Args:
        input_data: Comprehensive search parameters 
        onyx_service: Injected OnyxService instance (dependency injection)
        document_set_id: Document set ID for targeted Onyx search

    Returns:
        Dictionary containing unified results from multiple search systems
    """
    try:
        results = {
            "search_type": "comprehensive_hybrid_search",
            "query": input_data.query,
            "onyx_results": {},
            "vector_results": [],
            "graph_results": [],
            "combined_summary": "",
            "total_sources": 0,
            "systems_used": [],
            "primary_answer": "",
            "fallback_chain": [],
            "confidence": "low"
        }
        
        # Step 1: Try Onyx Cloud first (highest priority)
        onyx_success = False
        if input_data.include_onyx and onyx_service is not None:
            try:
                logger.info(f"Starting Onyx search for query: {input_data.query}")
                onyx_input = OnyxSearchInput(
                    query=input_data.query,
                    num_results=input_data.num_results,
                    search_type=input_data.search_type
                )
                onyx_result = await onyx_search_tool(onyx_input, onyx_service, document_set_id)
                results["onyx_results"] = onyx_result
                results["fallback_chain"].append("onyx_attempted")
                
                if onyx_result.get("success", False) and onyx_result.get("answer"):
                    onyx_success = True
                    results["systems_used"].append("Onyx Cloud")
                    results["total_sources"] += onyx_result.get("total_found", 0)
                    results["primary_answer"] = onyx_result["answer"]
                    results["confidence"] = onyx_result.get("confidence", "medium")
                    results["fallback_chain"].append("onyx_success")
                    
                    logger.info(f"✅ Onyx search successful, found {onyx_result.get('total_found', 0)} sources")
                    
                    # For comprehensive_search, ALWAYS continue to other systems for true synthesis
                    # No early exits - we want comprehensive multi-system analysis
                    logger.info("🔄 Continuing to Graphiti systems for comprehensive synthesis...")
                else:
                    logger.warning(f"❌ Onyx search failed or returned no results")
                    results["fallback_chain"].append("onyx_failed")
                        
            except Exception as e:
                logger.error(f"❌ Onyx search failed with exception: {e}")
                results["onyx_results"] = {"error": str(e), "success": False}
                results["fallback_chain"].append("onyx_error")
        else:
            if not input_data.include_onyx:
                results["fallback_chain"].append("onyx_disabled")
            else:
                results["fallback_chain"].append("onyx_unavailable")
        
        # Step 2: Graphiti fallback (vector + graph search)
        graphiti_tasks = []
        if input_data.include_vector:
            results["fallback_chain"].append("vector_search_started")
            graphiti_tasks.append(("vector", vector_search_tool(VectorSearchInput(
                query=input_data.query,
                limit=input_data.num_results
            ))))
        
        if input_data.include_graph:
            results["fallback_chain"].append("graph_search_started")
            graphiti_tasks.append(("graph", graph_search_tool(GraphSearchInput(query=input_data.query))))
        
        # Execute Graphiti searches (as fallback or complement)
        if graphiti_tasks:
            logger.info(f"Running Graphiti fallback with {len(graphiti_tasks)} search types")
            search_results = await asyncio.gather(
                *[task for _, task in graphiti_tasks],
                return_exceptions=True
            )
            
            # Process Graphiti results
            for i, (search_type, search_result) in enumerate(zip([t[0] for t in graphiti_tasks], search_results)):
                if isinstance(search_result, Exception):
                    logger.error(f"❌ {search_type} search failed: {search_result}")
                    results["fallback_chain"].append(f"{search_type}_error")
                    continue
                    
                if search_type == "vector" and search_result:
                    results["vector_results"] = search_result
                    results["systems_used"].append("Graphiti Vector")
                    results["total_sources"] += len(search_result)
                    results["fallback_chain"].append("vector_success")
                    logger.info(f"✅ Vector search found {len(search_result)} results")
                    
                elif search_type == "graph" and search_result:
                    results["graph_results"] = search_result
                    results["systems_used"].append("Graphiti Graph")
                    if hasattr(search_result, 'entities'):
                        results["total_sources"] += len(search_result.entities)
                    results["fallback_chain"].append("graph_success")
                    logger.info(f"✅ Graph search completed successfully")
        
        # Step 3: Intelligent Multi-System Synthesis
        if onyx_success and (results["vector_results"] or results["graph_results"]):
            # TRUE COMPREHENSIVE SYNTHESIS - Combine insights from all systems
            logger.info("🧠 Performing comprehensive multi-system synthesis...")
            
            # Start with Onyx as the factual foundation
            primary_content = results["onyx_results"]["answer"]
            
            # Add relationship context from knowledge graph
            relationship_insights = []
            if results["graph_results"]:
                for fact in results["graph_results"][:3]:  # Top 3 most relevant facts
                    if hasattr(fact, 'fact'):
                        relationship_insights.append(fact.fact)
                    elif isinstance(fact, dict) and "fact" in fact:
                        relationship_insights.append(fact["fact"])
                
                if relationship_insights:
                    results["fallback_chain"].append("graph_synthesis_added")
            
            # Add supporting evidence from vector search
            supporting_evidence = []
            if results["vector_results"]:
                for chunk in results["vector_results"][:2]:  # Top 2 supporting chunks
                    # ChunkResult objects use dot notation, not dictionary access
                    if hasattr(chunk, 'score') and chunk.score > 0.7:  # Only high-confidence chunks
                        content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                        supporting_evidence.append({
                            "content": content_preview,
                            "source": chunk.document_title if hasattr(chunk, 'document_title') else "Unknown",
                            "relevance": chunk.score
                        })
                
                if supporting_evidence:
                    results["fallback_chain"].append("vector_synthesis_added")
            
            # Create comprehensive synthesized answer
            synthesized_parts = [f"**Primary Answer (Onyx Cloud):** {primary_content}"]
            
            if relationship_insights:
                relationships_text = " | ".join(relationship_insights[:2])  # Top 2 relationships
                synthesized_parts.append(f"**Relationship Context (Knowledge Graph):** {relationships_text}")
            
            if supporting_evidence:
                evidence_text = f"Additional context from {len(supporting_evidence)} supporting sources"
                if supporting_evidence[0]:
                    evidence_text += f": {supporting_evidence[0]['content'][:150]}..."
                synthesized_parts.append(f"**Supporting Evidence (Vector Search):** {evidence_text}")
            
            # Create unified comprehensive response
            results["primary_answer"] = "\n\n".join(synthesized_parts)
            results["synthesis_type"] = "comprehensive_multi_system"
            results["confidence"] = "very_high"  # Multi-system validation
            results["fallback_chain"].append("multi_system_synthesis_complete")
            
        elif onyx_success:
            # Only Onyx succeeded - use as primary with high confidence
            results["synthesis_type"] = "onyx_primary"
            results["confidence"] = "high"
            results["fallback_chain"].append("onyx_only")
            
        elif results["vector_results"] or results["graph_results"]:
            # Onyx failed, use Graphiti results as primary
            if results["vector_results"]:
                # Use best vector result as primary answer
                best_chunk = max(results["vector_results"], key=lambda x: x.score if hasattr(x, 'score') else 0)
                content = best_chunk.content if hasattr(best_chunk, 'content') else str(best_chunk)
                results["primary_answer"] = f"**Vector Search Result:** {content[:500]}..."
                results["synthesis_type"] = "vector_primary"
                results["confidence"] = "medium"
                results["fallback_chain"].append("vector_primary")
                
                # Add graph context if available
                if results["graph_results"]:
                    graph_context = []
                    for fact in results["graph_results"][:2]:
                        if hasattr(fact, 'fact'):
                            graph_context.append(fact.fact)
                        elif isinstance(fact, dict) and "fact" in fact:
                            graph_context.append(fact["fact"])
                    
                    if graph_context:
                        results["primary_answer"] += f"\n\n**Related Knowledge:** {' | '.join(graph_context)}"
                        results["synthesis_type"] = "vector_graph_synthesis"
                        
            elif results["graph_results"]:
                # Use graph results summary 
                results["primary_answer"] = f"**Knowledge Graph Results:** Found related information for query: {input_data.query}"
                results["synthesis_type"] = "graph_primary"
                results["confidence"] = "low"
                results["fallback_chain"].append("graph_primary")
        else:
            # No results from any system
            results["synthesis_type"] = "no_results"
            results["confidence"] = "none"
                
        # Step 4: Generate combined summary
        summary_parts = []
        if onyx_success:
            summary_parts.append(f"Onyx Cloud: {results['onyx_results'].get('total_found', 0)} sources")
        if results["vector_results"]:
            summary_parts.append(f"Vector: {len(results['vector_results'])} chunks")
        if results["graph_results"]:
            summary_parts.append(f"Graph: relationships found")
            
        results["combined_summary"] = " | ".join(summary_parts) if summary_parts else "No results found"
        
        # Set final confidence based on what worked
        if onyx_success and results["total_sources"] > 0:
            results["confidence"] = "high"
        elif results["vector_results"] or results["graph_results"]:
            results["confidence"] = "medium" 
        else:
            results["confidence"] = "low"
            results["primary_answer"] = "No relevant information found across all search systems."
            
        logger.info(f"🎯 Comprehensive search completed: {results['combined_summary']}")
        logger.info(f"🔗 Fallback chain: {' -> '.join(results['fallback_chain'])}")
        
        return results
    
    except Exception as e:
        logger.error(f"Comprehensive search failed with exception: {e}")
        return {
            "search_type": "comprehensive_hybrid_search",
            "query": input_data.query,
            "onyx_results": {},
            "vector_results": [],
            "graph_results": [],
            "combined_summary": f"Search failed due to error: {str(e)}",
            "total_sources": 0,
            "systems_used": [],
            "primary_answer": f"Search error: {str(e)}",
            "fallback_chain": ["error"],
            "confidence": "none",
            "error": str(e)
        }
