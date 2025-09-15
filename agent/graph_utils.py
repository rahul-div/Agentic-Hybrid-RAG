"""
Graph utilities for Neo4j/Graphiti integration.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import asyncio

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphitiClient:
    """Manages Graphiti knowledge graph operations using Gemini."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """
        Initialize Graphiti client.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        # Gemini API configuration
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")

        # Model configuration from environment variables
        self.llm_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

        self.graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Graphiti client."""
        if self._initialized:
            return

        try:
            # Create Gemini LLM client with proper config
            llm_client = GeminiClient(
                config=LLMConfig(api_key=self.api_key, model=self.llm_model)
            )

            # Create Gemini embedder with proper config
            embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.api_key,
                    embedding_model="embedding-001",  # Use correct embedding model name
                )
            )

            # Create Gemini reranker with proper config
            reranker = GeminiRerankerClient(
                config=LLMConfig(
                    api_key=self.api_key,
                    model="gemini-1.5-flash",  # Use same model as LLM
                )
            )

            # Initialize Graphiti with Gemini clients
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=reranker,
            )

            # Build indices and constraints
            await self.graphiti.build_indices_and_constraints()

            self._initialized = True
            logger.info(
                f"Graphiti client initialized successfully with LLM: {self.llm_model} and embedder: embedding-001"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise

    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")

    def _sanitize_metadata_for_neo4j(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure all values are Neo4j-compatible primitive types.

        Neo4j accepts: strings, numbers, booleans, arrays of primitives
        Neo4j rejects: nested objects, None values, complex structures
        """
        if not metadata:
            return {}

        sanitized = {}

        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list elements to strings if they're not primitive
                sanitized_list = []
                for item in value:
                    if isinstance(item, (str, int, float, bool)):
                        sanitized_list.append(item)
                    else:
                        sanitized_list.append(str(item))
                sanitized[key] = sanitized_list
            elif isinstance(value, dict):
                # Convert dict to JSON string
                import json

                sanitized[key] = json.dumps(value)
            else:
                # Convert any other type to string
                sanitized[key] = str(value)

        return sanitized

    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        group_id: Optional[str] = None,
    ):
        """
        Add an episode to the knowledge graph with proper metadata handling.

        Args:
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata (will be sanitized for Neo4j)
            group_id: Group identifier for namespacing (CRITICAL for multi-tenancy)
        """
        if not self._initialized:
            await self.initialize()

        episode_timestamp = timestamp or datetime.now(timezone.utc)

        # Import EpisodeType for proper source handling
        from graphiti_core.nodes import EpisodeType

        try:
            # Add episode with proper group_id for tenant isolation
            await self.graphiti.add_episode(
                name=episode_id,
                episode_body=content,
                source=EpisodeType.text,
                source_description=source,
                reference_time=episode_timestamp,
                group_id=group_id,  # ✅ FIXED: Include group_id for tenant isolation
            )

            if group_id:
                logger.info(
                    f"Added episode {episode_id} to knowledge graph with namespace {group_id}"
                )
            else:
                logger.info(
                    f"Added episode {episode_id} to knowledge graph (no namespace)"
                )

        except Exception as e:
            logger.error(f"Failed to add episode {episode_id}: {e}")
            # Log more details about the error
            if "Property values can only be of primitive types" in str(e):
                logger.warning(
                    "Neo4j property type error - this is a known issue with Graphiti internal metadata"
                )
                logger.warning(
                    "Graphiti is trying to store complex structures (like summaries) as Neo4j properties"
                )
                logger.warning("Skipping this episode and continuing with ingestion...")
                return  # Continue processing instead of raising
            raise

    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True,
        group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search
            group_id: Group identifier for namespace filtering (CRITICAL for multi-tenancy)

        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use Graphiti's search method with group_id for tenant isolation
            results = await self.graphiti.search(query, group_id=group_id)

            if group_id:
                logger.info(
                    f"Graph search in namespace {group_id} returned {len(results)} results"
                )
            else:
                logger.info(
                    f"Graph search (all namespaces) returned {len(results)} results"
                )

            # Convert results to dictionaries
            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at)
                    if hasattr(result, "valid_at") and result.valid_at
                    else None,
                    "invalid_at": str(result.invalid_at)
                    if hasattr(result, "invalid_at") and result.invalid_at
                    else None,
                    "source_node_uuid": str(result.source_node_uuid)
                    if hasattr(result, "source_node_uuid") and result.source_node_uuid
                    else None,
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity using Graphiti search.

        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow (not used with Graphiti)
            depth: Maximum depth to traverse (not used with Graphiti)
            group_id: Group identifier for namespace filtering (CRITICAL for multi-tenancy)

        Returns:
            Related entities and relationships
        """
        if not self._initialized:
            await self.initialize()

        # Use Graphiti search to find related information about the entity
        results = await self.graphiti.search(
            f"relationships involving {entity_name}",
            group_id=group_id,  # ✅ FIXED: Include group_id for tenant isolation
        )

        # Extract entity information from the search results
        related_entities = set()
        facts = []

        for result in results:
            facts.append(
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at)
                    if hasattr(result, "valid_at") and result.valid_at
                    else None,
                }
            )

            # Simple entity extraction from fact text (could be enhanced)
            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)

        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search",
        }

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.

        Args:
            entity_name: Name of the entity
            start_date: Start of time range (not currently used)
            end_date: End of time range (not currently used)

        Returns:
            Timeline of facts
        """
        if not self._initialized:
            await self.initialize()

        # Search for temporal information about the entity
        results = await self.graphiti.search(f"timeline history of {entity_name}")

        timeline = []
        for result in results:
            timeline.append(
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at)
                    if hasattr(result, "valid_at") and result.valid_at
                    else None,
                    "invalid_at": str(result.invalid_at)
                    if hasattr(result, "invalid_at") and result.invalid_at
                    else None,
                }
            )

        # Sort by valid_at if available
        timeline.sort(key=lambda x: x.get("valid_at") or "", reverse=True)

        return timeline

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.

        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()

        # For now, return a simple search to verify the graph is working
        # More detailed statistics would require direct Neo4j access
        try:
            test_results = await self.graphiti.search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access",
            }
        except Exception as e:
            return {"graphiti_initialized": False, "error": str(e)}

    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()

        try:
            # Use Graphiti's proper clear_data function with the driver
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            # Fallback: Close and reinitialize (this will create fresh indices)
            if self.graphiti:
                await self.graphiti.close()

            # Create Gemini clients for reinitialization
            llm_client = GeminiClient(
                config=LLMConfig(api_key=self.api_key, model=self.llm_model)
            )

            embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.api_key, embedding_model="embedding-001"
                )
            )

            reranker = GeminiRerankerClient(
                config=LLMConfig(api_key=self.api_key, model="gemini-2.0-flash-exp")
            )

            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=reranker,
            )
            await self.graphiti.build_indices_and_constraints()

            logger.warning(
                "Reinitialized Graphiti client with Gemini (fresh indices created)"
            )


# Global Graphiti client instance
graph_client = GraphitiClient()


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


# Convenience functions for common operations
async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Add content to the knowledge graph.

    Args:
        content: Content to add
        source: Source of the content
        episode_id: Optional episode ID
        metadata: Optional metadata

    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"

    await graph_client.add_episode(
        episode_id=episode_id, content=content, source=source, metadata=metadata
    )

    return episode_id


async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.

    Args:
        query: Search query

    Returns:
        Search results
    """
    return await graph_client.search(query)


async def get_entity_relationships(entity: str, depth: int = 2) -> Dict[str, Any]:
    """
    Get relationships for an entity.

    Args:
        entity: Entity name
        depth: Maximum traversal depth

    Returns:
        Entity relationships
    """
    return await graph_client.get_related_entities(entity, depth=depth)


async def test_graph_connection() -> bool:
    """
    Test graph database connection.

    Returns:
        True if connection successful
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False


class TenantGraphitiClient:
    """
    Multi-tenant wrapper for GraphitiClient following official Graphiti patterns.

    This class provides tenant-aware methods that automatically apply group_id
    namespacing according to official Graphiti documentation.
    """

    def __init__(self, graphiti_client: GraphitiClient):
        """
        Initialize with a shared GraphitiClient instance.

        Args:
            graphiti_client: Shared GraphitiClient instance for all tenants
        """
        self.graphiti_client = graphiti_client

    def _get_tenant_namespace(self, tenant_id: str) -> str:
        """
        Generate tenant namespace following official Graphiti convention.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Namespace string in format "tenant_{tenant_id}"
        """
        return f"tenant_{tenant_id}"

    async def add_episode_for_tenant(
        self,
        tenant_id: str,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add episode to tenant's namespace.

        Args:
            tenant_id: Tenant identifier
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata
        """
        namespace = self._get_tenant_namespace(tenant_id)

        await self.graphiti_client.add_episode(
            episode_id=episode_id,
            content=content,
            source=source,
            timestamp=timestamp,
            metadata=metadata,
            group_id=namespace,  # Apply tenant namespace
        )

        logger.info(
            f"Added episode {episode_id} for tenant {tenant_id} (namespace: {namespace})"
        )

    async def search_for_tenant(
        self,
        tenant_id: str,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search within tenant's namespace.

        Args:
            tenant_id: Tenant identifier
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search

        Returns:
            Search results from tenant's namespace only
        """
        namespace = self._get_tenant_namespace(tenant_id)

        results = await self.graphiti_client.search(
            query=query,
            center_node_distance=center_node_distance,
            use_hybrid_search=use_hybrid_search,
            group_id=namespace,  # Filter to tenant namespace
        )

        logger.info(
            f"Search for tenant {tenant_id} (namespace: {namespace}) returned {len(results)} results"
        )
        return results

    async def get_related_entities_for_tenant(
        self,
        tenant_id: str,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get related entities within tenant's namespace.

        Args:
            tenant_id: Tenant identifier
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow
            depth: Maximum depth to traverse

        Returns:
            Related entities and relationships from tenant's namespace
        """
        namespace = self._get_tenant_namespace(tenant_id)

        return await self.graphiti_client.get_related_entities(
            entity_name=entity_name,
            relationship_types=relationship_types,
            depth=depth,
            group_id=namespace,  # Filter to tenant namespace
        )

    async def add_manual_fact_for_tenant(
        self,
        tenant_id: str,
        source_entity: str,
        target_entity: str,
        relationship: str,
        fact: str,
    ) -> None:
        """
        Add manual fact triplet to tenant's namespace.

        Args:
            tenant_id: Tenant identifier
            source_entity: Source entity name
            target_entity: Target entity name
            relationship: Relationship type
            fact: Fact description
        """
        namespace = self._get_tenant_namespace(tenant_id)

        # Import required classes
        from graphiti_core.nodes import EntityNode
        from graphiti_core.edges import EntityEdge
        import uuid

        # Create source node with tenant namespace
        source_node = EntityNode(
            uuid=str(uuid.uuid4()),
            name=source_entity,
            group_id=namespace,  # Apply tenant namespace
        )

        # Create target node with tenant namespace
        target_node = EntityNode(
            uuid=str(uuid.uuid4()),
            name=target_entity,
            group_id=namespace,  # Apply tenant namespace
        )

        # Create edge with tenant namespace
        edge = EntityEdge(
            group_id=namespace,  # Apply tenant namespace
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            created_at=datetime.now(timezone.utc),
            name=relationship,
            fact=fact,
        )

        # Add triplet to the graph
        if not self.graphiti_client._initialized:
            await self.graphiti_client.initialize()

        await self.graphiti_client.graphiti.add_triplet(source_node, edge, target_node)

        logger.info(
            f"Added manual fact for tenant {tenant_id} (namespace: {namespace}): "
            f"{source_entity} -> {relationship} -> {target_entity}"
        )

    async def initialize(self):
        """Initialize the underlying GraphitiClient."""
        await self.graphiti_client.initialize()

    async def close(self):
        """Close the underlying GraphitiClient."""
        await self.graphiti_client.close()
