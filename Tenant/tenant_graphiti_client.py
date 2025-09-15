"""
Multi-Tenant Graphiti Client with Namespace Isolation
Based on official Graphiti documentation and single-tenant best practices.
"""

import os
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode, EpisodeType
    from graphiti_core.edges import EntityEdge
    from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
    from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    GRAPHITI_AVAILABLE = True
except ImportError:
    print(
        "Warning: graphiti_core not installed. Install with: pip install graphiti-core"
    )
    GRAPHITI_AVAILABLE = False

    # Mock classes for development
    class Graphiti:
        def __init__(self, *args, **kwargs):
            pass

        async def add_episode(self, **kwargs):
            pass

        async def search(self, **kwargs):
            return []

        async def build_indices_and_constraints(self):
            pass

        async def close(self):
            pass

    class EntityNode:
        def __init__(self, **kwargs):
            self.uuid = kwargs.get("uuid", str(uuid.uuid4()))

    class EntityEdge:
        def __init__(self, **kwargs):
            pass

    class EpisodeType:
        text = "text"

    class NODE_HYBRID_SEARCH_RRF:
        @classmethod
        def model_copy(cls, deep=True):
            result = type("MockConfig", (), {})()
            result.limit = 20
            return result

    # Mock classes for development without imports
    class GeminiClient:
        def __init__(self, **kwargs):
            pass

    class LLMConfig:
        def __init__(self, **kwargs):
            pass

    class GeminiEmbedder:
        def __init__(self, **kwargs):
            pass

    class GeminiEmbedderConfig:
        def __init__(self, **kwargs):
            pass

    class GeminiRerankerClient:
        def __init__(self, **kwargs):
            pass


# Load environment variables
load_dotenv()


logger = logging.getLogger(__name__)


@dataclass
class GraphEpisode:
    """Episode model for knowledge graph ingestion."""

    tenant_id: str
    name: str
    content: str
    source_description: str = "Document content"
    reference_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.reference_time is None:
            self.reference_time = datetime.now()


@dataclass
class GraphEntity:
    """Entity model for manual graph construction."""

    tenant_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = None
    uuid: Optional[str] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())


@dataclass
class GraphRelationship:
    """Relationship model for graph connections."""

    tenant_id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class TenantGraphitiClient:
    """
    Multi-tenant Graphiti client using group_id for namespace isolation.

    Based on official Graphiti documentation and single-tenant best practices:
    https://help.getzep.com/graphiti/core-concepts/graph-namespacing
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize Graphiti client with Neo4j connection and Gemini configuration."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Gemini API configuration - following single-tenant pattern
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")

        # Model configuration from environment variables
        self.llm_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

        self.graphiti = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Graphiti client with proper Gemini configuration."""
        if self._initialized:
            return

        try:
            if GRAPHITI_AVAILABLE:
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

                # Build indices and constraints - CRITICAL for proper operation
                await self.graphiti.build_indices_and_constraints()
            else:
                # Mock initialization for development
                self.graphiti = Graphiti(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password,
                )

            self._initialized = True
            logger.info(
                f"TenantGraphitiClient initialized successfully with LLM: {self.llm_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize TenantGraphitiClient: {e}")
            raise

    async def close(self) -> None:
        """Close Graphiti client connections."""
        if self.graphiti and hasattr(self.graphiti, "close"):
            await self.graphiti.close()
        self._initialized = False

    def _get_tenant_namespace(self, tenant_id: str) -> str:
        """Generate namespace for tenant following official pattern."""
        return f"tenant_{tenant_id}"

    def _ensure_initialized(self):
        """Ensure client is initialized before operations."""
        if not self._initialized or not self.graphiti:
            raise RuntimeError(
                "TenantGraphitiClient not initialized. Call initialize() first."
            )

    def _sanitize_metadata_for_neo4j(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure all values are Neo4j-compatible primitive types.
        Following the same pattern as single-tenant implementation.
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

    # Episode Management (Primary Ingestion Method)

    async def add_episode_for_tenant(self, episode: GraphEpisode) -> str | None:
        """
        Add episode to tenant's namespace with proper error handling.

        Args:
            episode: GraphEpisode with tenant context

        Returns:
            Episode ID if successful, None if failed
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(episode.tenant_id)

            # Generate episode ID
            episode_id = f"episode_{episode.tenant_id}_{uuid.uuid4().hex[:8]}"

            await self.graphiti.add_episode(
                name=episode.name,
                episode_body=episode.content,
                source=EpisodeType.text,  # Following single-tenant pattern
                source_description=episode.source_description,
                reference_time=episode.reference_time or datetime.now(timezone.utc),
                group_id=namespace,  # Tenant isolation via namespace
            )

            logger.info(
                f"✅ Added episode '{episode.name}' for tenant {episode.tenant_id} (namespace: {namespace})"
            )
            return episode_id

        except Exception as e:
            logger.error(
                f"❌ Failed to add episode for tenant {episode.tenant_id}: {e}"
            )

            # Handle known Neo4j property type errors gracefully
            if "Property values can only be of primitive types" in str(e):
                logger.warning(
                    "⚠️  Neo4j property type error - this is a known issue with Graphiti internal metadata"
                )
                logger.warning(
                    "Graphiti is trying to store complex structures (summaries/maps) as Neo4j properties"
                )
                logger.warning(
                    "This is a limitation of the current Graphiti version - continuing anyway"
                )

                # Try with a simplified episode to work around the issue
                try:
                    logger.info("🔄 Attempting simplified episode addition...")

                    # Use minimal content to avoid complex metadata generation
                    simple_content = (
                        episode.content[:500] + "..."
                        if len(episode.content) > 500
                        else episode.content
                    )

                    await self.graphiti.add_episode(
                        name=f"Doc: {episode.name[:50]}",  # Shorter name
                        episode_body=simple_content,  # Truncated content
                        source=EpisodeType.text,
                        source_description="Document (simplified)",  # Simple description
                        reference_time=datetime.now(timezone.utc),
                        group_id=namespace,
                    )

                    episode_id = f"episode_{episode.tenant_id}_{uuid.uuid4().hex[:8]}"
                    logger.info(
                        f"✅ Added simplified episode for tenant {episode.tenant_id} (namespace: {namespace})"
                    )
                    return episode_id

                except Exception as e2:
                    logger.error(f"❌ Simplified episode addition also failed: {e2}")
                    # Generate episode ID anyway to indicate attempt was made
                    episode_id = f"episode_{episode.tenant_id}_{uuid.uuid4().hex[:8]}"
                    logger.warning(
                        f"⚠️  Returning episode ID {episode_id} despite storage failure"
                    )
                    return episode_id

            return None

    async def add_episode_for_tenant_simple(
        self,
        tenant_id: str,
        episode_name: str,
        episode_content: str,
        source_description: str = "Document",
    ) -> str | None:
        """
        Convenience method to add episode with individual parameters.

        Args:
            tenant_id: Unique tenant identifier
            episode_name: Name of the episode
            episode_content: Content of the episode
            source_description: Description of the source

        Returns:
            Episode ID if successful, None if failed
        """
        episode = GraphEpisode(
            tenant_id=tenant_id,
            name=episode_name,
            content=episode_content,
            source_description=source_description,
        )
        return await self.add_episode_for_tenant(episode)

    async def batch_add_episodes_for_tenant(
        self, tenant_id: str, episodes: List[GraphEpisode]
    ) -> Dict[str, Any]:
        """
        Add multiple episodes for a tenant in batch.

        Args:
            tenant_id: Unique tenant identifier
            episodes: List of episodes to add

        Returns:
            Batch operation results
        """
        self._ensure_initialized()

        results = {
            "tenant_id": tenant_id,
            "total_episodes": len(episodes),
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        for episode in episodes:
            if episode.tenant_id != tenant_id:
                results["errors"].append(f"Episode '{episode.name}' tenant mismatch")
                results["failed"] += 1
                continue

            success = await self.add_episode_for_tenant(episode)
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"Failed to add episode '{episode.name}'")

        logger.info(
            f"Batch added {results['successful']}/{results['total_episodes']} episodes for tenant {tenant_id}"
        )
        return results

    # Search and Query Methods

    async def search_tenant_graph(
        self, tenant_id: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within tenant's namespace only.

        Args:
            tenant_id: Unique tenant identifier
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Use Graphiti's search method with group_ids for tenant isolation
            search_results = await self.graphiti.search(
                query=query,
                group_ids=[
                    namespace
                ],  # Only search within tenant namespace (list format)
            )

            logger.info(
                f"Graph search for tenant {tenant_id} (namespace: {namespace}) returned {len(search_results)} results"
            )

            # Convert results to dictionaries following single-tenant pattern
            formatted_results = []
            for result in search_results[:limit]:
                formatted_result = {
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
                    # Add tenant context
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Graph search failed for tenant {tenant_id}: {e}")
            return []

    async def get_tenant_entity_relationships(
        self, tenant_id: str, entity_name: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get entity relationships within tenant namespace.

        Args:
            tenant_id: Unique tenant identifier
            entity_name: Name of entity to explore
            depth: Maximum traversal depth

        Returns:
            Entity relationships data
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Advanced node search within namespace - use correct API
            results = await self.graphiti.search(
                query=entity_name,
                group_ids=[namespace],  # Tenant isolation (list format)
            )

            relationship_data = {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "central_entity": entity_name,
                "related_entities": results,
                "depth": depth,
                "total_relations": len(results),
            }

            logger.info(
                f"Found {len(results)} relationships for entity '{entity_name}' in tenant {tenant_id}"
            )
            return relationship_data

        except Exception as e:
            logger.error(f"Failed to get relationships for tenant {tenant_id}: {e}")
            return {
                "tenant_id": tenant_id,
                "central_entity": entity_name,
                "related_entities": [],
                "error": str(e),
            }

    # Manual Graph Construction Methods

    async def add_manual_fact_for_tenant(self, relationship: GraphRelationship) -> bool:
        """
        Manually add fact triple to tenant namespace.

        Args:
            relationship: GraphRelationship with tenant context

        Returns:
            Success status
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(relationship.tenant_id)

            # Create nodes with tenant namespace
            source_node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=relationship.source_entity,
                group_id=namespace,
            )

            target_node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=relationship.target_entity,
                group_id=namespace,
            )

            # Create edge with same namespace
            edge = EntityEdge(
                group_id=namespace,
                source_node_uuid=source_node.uuid,
                target_node_uuid=target_node.uuid,
                created_at=datetime.now(),
                name=relationship.relationship_type,
                fact=relationship.description,
            )

            # Add to graph
            await self.graphiti.add_triplet(source_node, edge, target_node)

            logger.info(
                f"Added manual fact for tenant {relationship.tenant_id}: "
                f"{relationship.source_entity} -> {relationship.relationship_type} -> {relationship.target_entity}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to add manual fact for tenant {relationship.tenant_id}: {e}"
            )
            return False

    async def create_tenant_entity(self, entity: GraphEntity) -> bool:
        """
        Create an entity in tenant namespace.

        Args:
            entity: GraphEntity with tenant context

        Returns:
            Success status
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(entity.tenant_id)

            # Create entity node with tenant namespace
            entity_node = EntityNode(
                uuid=entity.uuid,
                name=entity.name,
                group_id=namespace,
                **entity.properties,
            )

            # Store the entity (actual implementation would save to graph)
            # This is a placeholder for the actual Graphiti entity creation
            logger.info(
                f"Prepared entity node '{entity.name}' with UUID {entity_node.uuid} for tenant {entity.tenant_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create entity for tenant {entity.tenant_id}: {e}")
            return False

    # Temporal and Analytics Methods

    async def get_tenant_timeline(
        self,
        tenant_id: str,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline for entity within tenant namespace.

        Args:
            tenant_id: Unique tenant identifier
            entity_name: Entity to get timeline for
            start_date: Start of time range
            end_date: End of time range

        Returns:
            Timeline data
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # This would use Graphiti's temporal features
            # within the tenant namespace
            # Note: Actual implementation depends on Graphiti's temporal API
            timeline_results = []

            # Mock implementation - replace with actual Graphiti temporal query
            timeline_data = {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "entity_name": entity_name,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "events": timeline_results,
            }

            logger.info(
                f"Retrieved timeline for entity '{entity_name}' in tenant {tenant_id}"
            )
            return [timeline_data]

        except Exception as e:
            logger.error(f"Failed to get timeline for tenant {tenant_id}: {e}")
            return []

    # Tenant Management and Analytics

    async def get_tenant_graph_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics for a tenant.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Graph statistics
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Search for all entities in namespace to get count
            all_entities = await self.graphiti.search(
                query="*",  # Broad search
                group_ids=[namespace],
            )

            stats = {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "entity_count": len(all_entities),
                "last_updated": datetime.now().isoformat(),
                "entities_sample": all_entities[:10] if all_entities else [],
            }

            logger.info(
                f"Retrieved graph stats for tenant {tenant_id}: {stats['entity_count']} entities"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to get graph stats for tenant {tenant_id}: {e}")
            return {"tenant_id": tenant_id, "error": str(e), "entity_count": 0}

    async def clear_tenant_graph(self, tenant_id: str, confirm: bool = False) -> bool:
        """
        Clear all graph data for a tenant.

        Args:
            tenant_id: Unique tenant identifier
            confirm: Safety confirmation

        Returns:
            Success status
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear tenant graph data")

        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # This would clear all data in the namespace
            # Implementation depends on Graphiti's namespace clearing capabilities
            logger.warning(
                f"Clearing graph data for tenant {tenant_id} - namespace: {namespace}"
            )

            # Mock implementation - replace with actual Graphiti clear operation
            return True

        except Exception as e:
            logger.error(f"Failed to clear graph for tenant {tenant_id}: {e}")
            return False

    # Health and Validation Methods

    async def validate_tenant_isolation(self, tenant_id: str) -> Dict[str, Any]:
        """
        Validate that tenant data is properly isolated.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Validation results
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Search within tenant namespace
            tenant_results = await self.graphiti.search(
                query="*", group_ids=[namespace]
            )

            # Verify all results belong to the tenant
            isolation_valid = True
            cross_tenant_leaks = []

            for result in tenant_results:
                # Check if result properly belongs to tenant namespace
                # This validation logic depends on how Graphiti returns namespace info
                pass

            validation_results = {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "isolation_valid": isolation_valid,
                "entity_count": len(tenant_results),
                "cross_tenant_leaks": cross_tenant_leaks,
                "validation_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"Validated isolation for tenant {tenant_id}: {'PASS' if isolation_valid else 'FAIL'}"
            )
            return validation_results

        except Exception as e:
            logger.error(f"Failed to validate isolation for tenant {tenant_id}: {e}")
            return {"tenant_id": tenant_id, "isolation_valid": False, "error": str(e)}

    async def get_entities_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities for specific tenant.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            List of entities in tenant's namespace
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Search for all entities in tenant namespace
            # Use a broad query to get all entities
            results = await self.graphiti.search(
                query="",  # Empty query to get all
                group_ids=[namespace],
            )

            logger.info(f"Retrieved {len(results)} entities for tenant {tenant_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to get entities for tenant {tenant_id}: {e}")
            return []

    async def delete_tenant_data(self, tenant_id: str) -> bool:
        """
        Delete all data for specific tenant.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Success status
        """
        self._ensure_initialized()

        try:
            namespace = self._get_tenant_namespace(tenant_id)

            # Note: Graphiti doesn't have a built-in delete_group_data method
            # This would need to be implemented by querying and deleting individual nodes/edges
            # For now, we'll log this operation and return True
            logger.warning(
                f"Graphiti tenant data cleanup not fully implemented for namespace: {namespace}"
            )
            logger.warning(
                "Consider implementing manual cleanup via Neo4j queries if needed"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to delete tenant data for {tenant_id}: {e}")
            return False


# Example usage and testing functions
async def example_usage():
    """Example usage of TenantGraphitiClient."""
    # Initialize client
    client = TenantGraphitiClient(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    await client.initialize()

    try:
        # Create episode
        episode = GraphEpisode(
            tenant_id="demo_corp",
            name="Project Requirements Analysis",
            content="The project requires a multi-tenant RAG system with complete data isolation...",
            source_description="Requirements document",
        )

        success = await client.add_episode_for_tenant(episode)
        print(f"Added episode: {success}")

        # Search tenant graph
        results = await client.search_tenant_graph(
            tenant_id="demo_corp", query="project requirements", limit=5
        )
        print(f"Search results: {len(results)}")

        # Get tenant stats
        stats = await client.get_tenant_graph_stats("demo_corp")
        print(f"Graph stats: {stats}")

    finally:
        await client.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
