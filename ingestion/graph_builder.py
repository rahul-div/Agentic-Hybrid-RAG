"""
Knowledge graph builder for extracting entities and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import re

from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks using Gemini AI models."""

    def __init__(self):
        """Initialize graph builder for Gemini-powered knowledge graphs."""
        self.graph_client = GraphitiClient()
        self._initialized = False
        logger.info("GraphBuilder initialized for Gemini knowledge graph processing")

    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True

    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3,  # Reduced batch size for Graphiti
        group_id: Optional[str] = None,  # For client namespacing
    ) -> Dict[str, Any]:
        """
        Add document chunks to the knowledge graph.

        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata
            batch_size: Number of chunks to process in each batch

        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return {"episodes_created": 0, "errors": []}

        logger.info(
            f"Adding {len(chunks)} chunks to Gemini-powered knowledge graph for document: {document_title}"
        )
        logger.info(
            "⚠️ Large chunks will be truncated to avoid Gemini/Graphiti token limits."
        )

        # Check for oversized chunks and warn
        oversized_chunks = [
            i for i, chunk in enumerate(chunks) if len(chunk.content) > 6000
        ]
        if oversized_chunks:
            logger.warning(
                f"Found {len(oversized_chunks)} chunks over 6000 chars that will be truncated: {oversized_chunks}"
            )

        episodes_created = 0
        errors = []

        # Process chunks one by one to avoid overwhelming Graphiti
        for i, chunk in enumerate(chunks):
            try:
                # Create episode ID
                episode_id = (
                    f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"
                )

                # Prepare episode content with size limits
                episode_content = self._prepare_episode_content(
                    chunk, document_title, document_metadata
                )

                # Create source description (shorter)
                source_description = (
                    f"Document: {document_title} (Chunk: {chunk.index})"
                )

                # Add episode to graph - flatten metadata for Graphiti compatibility
                flattened_metadata = {
                    "document_title": document_title,
                    "document_source": document_source,
                    "chunk_index": chunk.index,
                    "original_length": len(chunk.content),
                    "processed_length": len(episode_content),
                }

                # Flatten entity information for graph compatibility
                if hasattr(chunk, "metadata") and "entities" in chunk.metadata:
                    entities = chunk.metadata["entities"]
                    for entity_type, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            # Convert list to comma-separated string for graph storage
                            flattened_metadata[f"entities_{entity_type}"] = ", ".join(
                                entity_list
                            )

                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata=flattened_metadata,
                    group_id=group_id,  # Pass group_id for namespacing
                )

                episodes_created += 1
                logger.info(
                    f"✓ Added episode {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})"
                )

                # Small delay between each episode to reduce API pressure
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Continue processing other chunks even if one fails
                continue

        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors,
        }

        logger.info(
            f"Graph building complete: {episodes_created} episodes created, {len(errors)} errors"
        )
        return result

    def _prepare_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Prepare episode content with minimal context to avoid token limits.

        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata

        Returns:
            Formatted episode content (optimized for Gemini/Graphiti)
        """
        # Limit chunk content to avoid Gemini/Graphiti's token limits
        # Estimate ~4 chars per token, keep content under 6000 chars for Gemini processing
        max_content_length = 6000

        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? ")
            )

            if (
                last_sentence_end > max_content_length * 0.7
            ):  # If we can keep 70% and end cleanly
                content = truncated[: last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"

            logger.warning(
                f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars for Gemini/Graphiti"
            )

        # Add minimal context (just document title for now)
        if document_title and len(content) < max_content_length - 100:
            episode_content = f"[Doc: {document_title[:50]}]\n\n{content}"
        else:
            episode_content = content

        return episode_content

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)."""
        return len(text) // 4

    def _is_content_too_large(self, content: str, max_tokens: int = 7000) -> bool:
        """Check if content is too large for Gemini/Graphiti processing."""
        return self._estimate_tokens(content) > max_tokens

    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_clients: bool = True,
        extract_projects: bool = True,
        extract_requirements: bool = True,
        extract_tasks: bool = True,
        extract_team_members: bool = True,
        extract_technologies: bool = True,
        extract_people: bool = False,  # Legacy support, disabled by default
    ) -> List[DocumentChunk]:
        """
        Extract project management entities from chunks and add to metadata.

        Args:
            chunks: List of document chunks
            extract_clients: Whether to extract client names
            extract_projects: Whether to extract project names
            extract_requirements: Whether to extract requirements
            extract_tasks: Whether to extract tasks and epics
            extract_team_members: Whether to extract team member names
            extract_technologies: Whether to extract technology terms
            extract_people: Whether to extract general people names (legacy)

        Returns:
            Chunks with project management entity metadata added
        """
        logger.info(
            f"Extracting project management entities from {len(chunks)} chunks using rule-based extraction (optimized for Gemini processing)"
        )

        enriched_chunks = []

        for chunk in chunks:
            entities = {
                "clients": [],
                "projects": [],
                "requirements": [],
                "tasks": [],
                "team_members": [],
                "technologies": [],
                "people": [],  # Legacy support
                "locations": [],
            }

            content = chunk.content

            # Extract clients
            if extract_clients:
                entities["clients"] = self._extract_clients(content)

            # Extract projects
            if extract_projects:
                entities["projects"] = self._extract_projects(content)

            # Extract requirements
            if extract_requirements:
                entities["requirements"] = self._extract_requirements(content)

            # Extract tasks
            if extract_tasks:
                entities["tasks"] = self._extract_tasks(content)

            # Extract team members
            if extract_team_members:
                entities["team_members"] = self._extract_team_members(content)

            # Extract technologies
            if extract_technologies:
                entities["technologies"] = self._extract_technologies(content)

            # Extract people (legacy support)
            if extract_people:
                entities["people"] = self._extract_people(content)

            # Extract locations
            entities["locations"] = self._extract_locations(content)

            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entities": entities,
                    "entity_extraction_date": datetime.now().isoformat(),
                },
                token_count=chunk.token_count,
            )

            # Preserve embedding if it exists
            if hasattr(chunk, "embedding"):
                enriched_chunk.embedding = chunk.embedding

            enriched_chunks.append(enriched_chunk)

        logger.info("Entity extraction complete")
        return enriched_chunks

    def _extract_clients(self, text: str) -> List[str]:
        """Extract client/company names from project documents."""
        # Common patterns for client identification
        client_patterns = [
            r"client[:\s]+([A-Z][a-zA-Z\s&.]+(?:Inc|LLC|Corp|Ltd|Limited|Company|Co\.)?)",
            r"customer[:\s]+([A-Z][a-zA-Z\s&.]+(?:Inc|LLC|Corp|Ltd|Limited|Company|Co\.)?)",
            r"for\s+([A-Z][a-zA-Z\s&.]+(?:Inc|LLC|Corp|Ltd|Limited|Company|Co\.))",
            r"([A-Z][a-zA-Z\s&.]+(?:Inc|LLC|Corp|Ltd|Limited|Company|Co\.))\s+project",
        ]

        found_clients = set()
        text_lines = text.split("\n")

        for line in text_lines:
            for pattern in client_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    client = match.strip().strip(":").strip()
                    if len(client) > 2 and len(client) < 100:  # Reasonable length
                        found_clients.add(client)

        return list(found_clients)

    def _extract_projects(self, text: str) -> List[str]:
        """Extract project names and codes from text."""
        project_patterns = [
            r"project[:\s]+([A-Z][a-zA-Z\s\-_]+)",
            r"([A-Z]{2,}-[0-9]+)",  # Project codes like "PROJ-123"
            r"([A-Z]{3,}[0-9]{2,})",  # Project codes like "ABC123"
            r"project\s+name[:\s]+([A-Za-z\s\-_]+)",
            r"([A-Za-z\s]+)\s+project",
        ]

        found_projects = set()

        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                project = match.strip().strip(":").strip()
                if len(project) > 2 and len(project) < 100:
                    found_projects.add(project)

        return list(found_projects)

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract business and technical requirements."""
        requirement_keywords = {
            "user authentication",
            "user management",
            "dashboard",
            "reporting",
            "api integration",
            "database",
            "payment processing",
            "notification system",
            "user interface",
            "mobile app",
            "web application",
            "data analytics",
            "security",
            "backup",
            "performance",
            "scalability",
            "integration",
            "workflow",
            "approval process",
            "role-based access",
            "single sign-on",
        }

        found_requirements = set()
        text_lower = text.lower()

        for requirement in requirement_keywords:
            if requirement in text_lower:
                found_requirements.add(requirement.title())

        # Also look for requirements patterns
        req_patterns = [
            r"requirement[:\s]+([A-Za-z\s\-_]+)",
            r"must\s+([a-z\s]+)",
            r"should\s+([a-z\s]+)",
            r"feature[:\s]+([A-Za-z\s\-_]+)",
        ]

        for pattern in req_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                req = match.strip().strip(":").strip()
                if len(req) > 5 and len(req) < 100:
                    found_requirements.add(req.title())

        return list(found_requirements)

    def _extract_tasks(self, text: str) -> List[str]:
        """Extract tasks, epics, and user stories."""
        task_patterns = [
            r"epic[:\s]+([A-Za-z\s\-_]+)",
            r"user\s+story[:\s]+([A-Za-z\s\-_]+)",
            r"task[:\s]+([A-Za-z\s\-_]+)",
            r"subtask[:\s]+([A-Za-z\s\-_]+)",
            r"ticket[:\s]+([A-Z]+-[0-9]+)",
            r"([A-Z]+-[0-9]+)[:\s]+([A-Za-z\s\-_]+)",  # JIRA-style tickets
        ]

        found_tasks = set()

        for pattern in task_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    task = " ".join(match).strip()
                else:
                    task = match.strip().strip(":").strip()
                if len(task) > 3 and len(task) < 150:
                    found_tasks.add(task)

        return list(found_tasks)

    def _extract_team_members(self, text: str) -> List[str]:
        """Extract team member names and roles."""
        role_patterns = [
            r"project\s+manager[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"developer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"designer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"analyst[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"lead[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"assigned\s+to[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]

        found_members = set()

        for pattern in role_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                member = match.strip().strip(":").strip()
                if len(member) > 3 and len(member) < 50:
                    found_members.add(member)

        return list(found_members)

    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technologies and tools mentioned in project documents."""
        technologies = {
            # Programming Languages
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "C#",
            "Go",
            "Rust",
            "PHP",
            "Ruby",
            "Swift",
            "Kotlin",
            "Dart",
            # Frameworks & Libraries
            "React",
            "Angular",
            "Vue.js",
            "Node.js",
            "Express",
            "Django",
            "Flask",
            "Spring Boot",
            "Laravel",
            "Ruby on Rails",
            ".NET",
            "Flutter",
            # Databases
            "PostgreSQL",
            "MySQL",
            "MongoDB",
            "Redis",
            "Elasticsearch",
            "SQLite",
            "Oracle",
            "SQL Server",
            "DynamoDB",
            "Cassandra",
            # Cloud & Infrastructure
            "AWS",
            "Azure",
            "Google Cloud",
            "Docker",
            "Kubernetes",
            "Terraform",
            "Jenkins",
            "GitLab CI",
            "GitHub Actions",
            # Tools
            "Jira",
            "Confluence",
            "Slack",
            "Teams",
            "Figma",
            "Adobe XD",
            "Postman",
            "Swagger",
            "Git",
            "VS Code",
            "IntelliJ",
        }

        found_technologies = set()
        text_lower = text.lower()

        for tech in technologies:
            pattern = r"\b" + re.escape(tech.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found_technologies.add(tech)

        return list(found_technologies)
        """Extract technology terms from text."""
        tech_terms = {
            "AI",
            "artificial intelligence",
            "machine learning",
            "ML",
            "deep learning",
            "neural network",
            "LLM",
            "large language model",
            "GPT",
            "transformer",
            "NLP",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "generative AI",
            "foundation model",
            "multimodal",
            "chatbot",
            "API",
            "cloud computing",
            "edge computing",
            "quantum computing",
            "blockchain",
            "cryptocurrency",
            "IoT",
            "5G",
            "AR",
            "VR",
            "autonomous vehicles",
            "robotics",
            "automation",
            # Add Gemini-specific terms
            "Gemini",
            "Bard",
            "PaLM",
            "LaMDA",
            "embedding-001",
        }

        found_terms = set()
        text_lower = text.lower()

        for term in tech_terms:
            if term.lower() in text_lower:
                found_terms.add(term)

        return list(found_terms)

    def _extract_people(self, text: str) -> List[str]:
        """Extract person names from text."""
        # Known tech leaders (extend this list as needed)
        tech_leaders = {
            "Elon Musk",
            "Jeff Bezos",
            "Tim Cook",
            "Satya Nadella",
            "Sundar Pichai",
            "Mark Zuckerberg",
            "Sam Altman",
            "Dario Amodei",
            "Daniela Amodei",
            "Jensen Huang",
            "Bill Gates",
            "Larry Page",
            "Sergey Brin",
            "Jack Dorsey",
            "Reed Hastings",
            "Marc Benioff",
            "Andy Jassy",
        }

        found_people = set()

        for person in tech_leaders:
            if person in text:
                found_people.add(person)

        return list(found_people)

    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        locations = {
            "Silicon Valley",
            "San Francisco",
            "Seattle",
            "Austin",
            "New York",
            "Boston",
            "London",
            "Tel Aviv",
            "Singapore",
            "Beijing",
            "Shanghai",
            "Tokyo",
            "Seoul",
            "Bangalore",
            "Mountain View",
            "Cupertino",
            "Redmond",
            "Menlo Park",
        }

        found_locations = set()

        for location in locations:
            if location in text:
                found_locations.add(location)

        return list(found_locations)

    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        logger.warning("Clearing Gemini-powered knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Gemini-powered knowledge graph cleared")


class SimpleEntityExtractor:
    """Simple rule-based entity extractor as fallback."""

    def __init__(self):
        """Initialize extractor."""
        self.company_patterns = [
            r"\b(?:Google|Microsoft|Apple|Amazon|Meta|Facebook|Tesla|OpenAI)\b",
            r"\b\w+\s+(?:Inc|Corp|Corporation|Ltd|Limited|AG|SE)\b",
        ]

        self.tech_patterns = [
            r"\b(?:AI|artificial intelligence|machine learning|ML|deep learning)\b",
            r"\b(?:neural network|transformer|GPT|LLM|NLP)\b",
            r"\b(?:cloud computing|API|blockchain|IoT|5G)\b",
            r"\b(?:Gemini|Bard|PaLM|LaMDA|embedding-001)\b",  # Gemini-specific terms
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using patterns."""
        entities = {"companies": [], "technologies": []}

        # Extract companies
        for pattern in self.company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["companies"].extend(matches)

        # Extract technologies
        for pattern in self.tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["technologies"].extend(matches)

        # Remove duplicates and clean up
        entities["companies"] = list(set(entities["companies"]))
        entities["technologies"] = list(set(entities["technologies"]))

        return entities


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create Gemini-powered graph builder instance."""
    return GraphBuilder()


# Example usage
async def main():
    """Example usage of the graph builder."""
    from .chunker import ChunkingConfig, create_chunker

    # Create chunker and graph builder
    config = ChunkingConfig(chunk_size=300, use_semantic_splitting=False)
    chunker = create_chunker(config)
    graph_builder = create_graph_builder()

    sample_text = """
    Google's DeepMind has made significant breakthroughs in artificial intelligence,
    particularly in areas like protein folding prediction with AlphaFold and
    game-playing AI with AlphaGo. The company continues to invest heavily in
    transformer architectures and large language models.
    
    Microsoft's partnership with OpenAI has positioned them as a leader in
    the generative AI space. Sam Altman's OpenAI has developed GPT models
    that Microsoft integrates into Office 365 and Azure cloud services.
    """

    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text, title="AI Company Developments", source="example.md"
    )

    print(f"Created {len(chunks)} chunks")

    # Extract entities
    enriched_chunks = await graph_builder.extract_entities_from_chunks(chunks)

    for i, chunk in enumerate(enriched_chunks):
        print(f"Chunk {i}: {chunk.metadata.get('entities', {})}")

    # Add to knowledge graph
    try:
        result = await graph_builder.add_document_to_graph(
            chunks=enriched_chunks,
            document_title="AI Company Developments",
            document_source="example.md",
            document_metadata={"topic": "AI", "date": "2024"},
        )

        print(f"Graph building result: {result}")

    except Exception as e:
        print(f"Graph building failed: {e}")

    finally:
        await graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())
