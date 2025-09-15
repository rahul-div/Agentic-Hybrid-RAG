"""
Main ingestion script for processing markdown documents into vector DB and knowledge graph.
"""

import os
import asyncio
import logging
import json
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

from dotenv import load_dotenv

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .graph_builder import create_graph_builder
from .onyx_ingest import ingest_to_onyx, OnyxIngestionError

# Import agent utilities
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult

# Load environment variables
load_dotenv()

# Load .env from project root as well (backup)
project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

logger = logging.getLogger(__name__)


def validate_gemini_configuration():
    """Validate that Gemini configuration is properly set up."""
    # Debug: Check if .env file is being loaded
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    logger.info(f"Looking for .env file at: {env_path}")
    logger.info(f".env file exists: {os.path.exists(env_path)}")

    # Check for required API keys with debug info
    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    logger.info(f"GOOGLE_API_KEY found: {bool(google_api_key)}")
    logger.info(f"GEMINI_API_KEY found: {bool(gemini_api_key)}")

    if google_api_key:
        logger.info(f"GOOGLE_API_KEY starts with: {google_api_key[:10]}...")
    if gemini_api_key:
        logger.info(f"GEMINI_API_KEY starts with: {gemini_api_key[:10]}...")

    api_key = google_api_key or gemini_api_key
    if not api_key:
        logger.error("No Gemini API key found in environment variables")
        logger.error(
            "Please check that your .env file contains GOOGLE_API_KEY or GEMINI_API_KEY"
        )
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY must be set for Gemini configuration"
        )

    # Check embedding provider
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if embedding_provider not in ["gemini", ""]:
        logger.warning(
            f"EMBEDDING_PROVIDER is set to '{embedding_provider}' but should be 'gemini' for consistent configuration"
        )

    # Check vector dimensions
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    if vector_dimension != 768:
        logger.warning(
            f"VECTOR_DIMENSION is set to {vector_dimension} but should be 768 for Gemini embeddings"
        )

    # Check embedding model
    embedding_model = os.getenv("EMBEDDING_MODEL", "")
    if embedding_model and embedding_model != "embedding-001":
        logger.warning(
            f"EMBEDDING_MODEL is set to '{embedding_model}' but should be 'embedding-001' for Gemini"
        )

    logger.info("Gemini configuration validated successfully")
    return api_key


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector DB and knowledge graph."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            documents_folder: Folder containing markdown documents
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()

        self._initialized = False

    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")

        # Validate Gemini configuration
        validate_gemini_configuration()

        # Initialize database connections
        await initialize_database()
        await initialize_graph()
        await self.graph_builder.initialize()

        self._initialized = True
        logger.info("Ingestion pipeline initialized with Gemini configuration")

    async def close(self):
        """Close database connections."""
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_database()
            self._initialized = False

    async def ingest_documents(
        self, progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()

        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases()

        # Find all markdown files
        markdown_files = self._find_markdown_files()

        if not markdown_files:
            logger.warning(f"No markdown files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(markdown_files)} markdown files to process")

        results = []

        for i, file_path in enumerate(markdown_files):
            try:
                logger.info(
                    f"Processing file {i + 1}/{len(markdown_files)}: {file_path}"
                )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(markdown_files))

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        entities_extracted=0,
                        relationships_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(
            f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors"
        )

        return results

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to the document file

        Returns:
            Ingestion result
        """
        start_time = datetime.now()

        # Read document
        document_content = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)

        # Extract metadata from content
        document_metadata = self._extract_document_metadata(document_content, file_path)

        logger.info(f"Processing document: {document_title}")

        # EPIC 3: Onyx Cloud ingestion (before Graphiti processing)
        onyx_result = {
            "ingested": False,
            "document_id": None,
            "sections_count": 0,
            "errors": [],
        }

        if self.config.enable_onyx_ingestion:
            logger.info(f"🔮 Starting Onyx Cloud dual ingestion for: {document_title}")
            try:
                onyx_ingestion_result = await ingest_to_onyx(
                    content=document_content,
                    file_path=file_path,
                    metadata={
                        **document_metadata,
                        "dual_ingestion": True,
                        "local_document_title": document_title,
                        "local_document_source": document_source,
                    },
                )

                onyx_result = {
                    "ingested": onyx_ingestion_result.get("success", False),
                    "document_id": onyx_ingestion_result.get("document_id"),
                    "sections_count": onyx_ingestion_result.get("sections_count", 0),
                    "errors": [],
                }

                logger.info(
                    f"✅ Onyx Cloud ingestion completed: {onyx_result['document_id']} ({onyx_result['sections_count']} sections)"
                )

            except OnyxIngestionError as e:
                error_msg = f"Onyx ingestion failed: {str(e)}"
                logger.warning(f"⚠️  {error_msg}")
                onyx_result["errors"].append(error_msg)

            except Exception as e:
                error_msg = f"Unexpected Onyx ingestion error: {str(e)}"
                logger.error(f"❌ {error_msg}")
                onyx_result["errors"].append(error_msg)
        else:
            logger.debug("Onyx Cloud ingestion disabled in configuration")

        # Chunk the document
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"],
            )

        logger.info(f"Created {len(chunks)} chunks")

        # Extract entities if configured
        entities_extracted = 0
        if self.config.extract_entities:
            chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
            entities_extracted = sum(
                len(chunk.metadata.get("entities", {}).get("clients", []))
                + len(chunk.metadata.get("entities", {}).get("projects", []))
                + len(chunk.metadata.get("entities", {}).get("requirements", []))
                + len(chunk.metadata.get("entities", {}).get("tasks", []))
                + len(chunk.metadata.get("entities", {}).get("team_members", []))
                + len(chunk.metadata.get("entities", {}).get("technologies", []))
                + len(
                    chunk.metadata.get("entities", {}).get("people", [])
                )  # Legacy support
                + len(chunk.metadata.get("entities", {}).get("locations", []))
                for chunk in chunks
            )
            logger.info(f"Extracted {entities_extracted} project management entities")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(
            f"Generated embeddings for {len(embedded_chunks)} chunks using Gemini embedding-001 (768 dimensions)"
        )

        # Validate embedding dimensions for Gemini
        for i, chunk in enumerate(embedded_chunks):
            if hasattr(chunk, "embedding") and chunk.embedding:
                if len(chunk.embedding) != 768:
                    logger.error(
                        f"Chunk {i} has {len(chunk.embedding)} dimensions, expected 768 for Gemini"
                    )
                    raise ValueError(
                        f"Invalid embedding dimensions for Gemini: expected 768, got {len(chunk.embedding)}"
                    )

        # Save to PostgreSQL
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
        )

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        # Add to knowledge graph (if enabled)
        relationships_created = 0
        graph_errors = []

        if not self.config.skip_graph_building:
            try:
                logger.info(
                    "Building knowledge graph relationships (this may take several minutes)..."
                )
                graph_result = await self.graph_builder.add_document_to_graph(
                    chunks=embedded_chunks,
                    document_title=document_title,
                    document_source=document_source,
                    document_metadata=document_metadata,
                )

                relationships_created = graph_result.get("episodes_created", 0)
                graph_errors = graph_result.get("errors", [])

                logger.info(
                    f"Added {relationships_created} episodes to knowledge graph"
                )

            except Exception as e:
                error_msg = f"Failed to add to knowledge graph: {str(e)}"
                logger.error(error_msg)
                graph_errors.append(error_msg)
        else:
            logger.info("Skipping knowledge graph building (skip_graph_building=True)")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Combine all errors (graph + onyx)
        all_errors = graph_errors + onyx_result.get("errors", [])

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
            processing_time_ms=processing_time,
            errors=all_errors,
            # EPIC 3: Onyx Cloud ingestion results
            onyx_ingested=onyx_result.get("ingested", False),
            onyx_document_id=onyx_result.get("document_id"),
            onyx_sections_count=onyx_result.get("sections_count", 0),
        )

    def _find_markdown_files(self) -> List[str]:
        """Find markdown files in the documents folder."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        patterns = ["*.md", "*.markdown", "*.txt", "*.yaml", "*.yml"]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern), recursive=True
                )
            )

        all_files = sorted(files)

        # Check if single document mode is enabled
        if getattr(self.config, "single_document_mode", False):
            if all_files:
                first_file = [all_files[0]]  # Only process the first document
                logger.info(
                    f"🔥 SINGLE DOCUMENT MODE: Processing only first document: {os.path.basename(first_file[0])}"
                )
                logger.info(
                    f"📋 Skipping {len(all_files) - 1} other documents (use --all-docs to process all)"
                )
                return first_file
        else:
            if all_files:
                logger.info(f"📚 FULL MODE: Processing all {len(all_files)} documents")

        return all_files

    def _read_document(self, file_path: str) -> str:
        """Read document content from file, handling different formats."""
        try:
            # Check if it's a YAML file
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in [".yaml", ".yml"]:
                return self._read_yaml_file(file_path)
            else:
                # Read as regular text file
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _read_yaml_file(self, file_path: str) -> str:
        """Read and convert YAML file to searchable text format."""
        try:
            import yaml

            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)

            # Convert YAML to readable text format
            return self._yaml_to_text(yaml_content, file_path)

        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install PyYAML")
            raise
        except Exception as e:
            logger.error(f"Failed to read YAML file {file_path}: {e}")
            # Fallback to reading as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def _yaml_to_text(self, yaml_content: Any, file_path: str) -> str:
        """Convert YAML content to readable text format for indexing."""

        def format_value(value, indent=0):
            """Recursively format YAML values as readable text."""
            spaces = "  " * indent

            if isinstance(value, dict):
                result = []
                for key, val in value.items():
                    if isinstance(val, (dict, list)):
                        result.append(f"{spaces}{key}:")
                        result.append(format_value(val, indent + 1))
                    else:
                        result.append(f"{spaces}{key}: {val}")
                return "\n".join(result)

            elif isinstance(value, list):
                result = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        result.append(format_value(item, indent))
                    else:
                        result.append(f"{spaces}- {item}")
                return "\n".join(result)

            else:
                return f"{spaces}{value}"

        # Create a readable text representation
        filename = os.path.basename(file_path)
        text_parts = [f"# {filename}", ""]

        if isinstance(yaml_content, dict):
            text_parts.append(format_value(yaml_content))
        elif isinstance(yaml_content, list):
            text_parts.append("Document Contents:")
            text_parts.append(format_value(yaml_content))
        else:
            text_parts.append(f"Content: {yaml_content}")

        return "\n".join(text_parts)

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        # Try to find markdown title
        lines = content.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(
        self, content: str, file_path: str
    ) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
        }

        # Try to extract YAML frontmatter
        if content.startswith("---"):
            try:
                import yaml

                end_marker = content.find("\n---\n", 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning("PyYAML not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

        # Extract some basic metadata from content
        lines = content.split("\n")
        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(content.split())

        return metadata

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        """Save document and chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata),
                )

                document_id = document_result["id"]

                # Insert chunks
                for chunk in chunks:
                    # Convert embedding to PostgreSQL vector string format
                    embedding_data = None
                    if hasattr(chunk, "embedding") and chunk.embedding:
                        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
                        # Ensure we have exactly 768 dimensions for Gemini embeddings
                        if len(chunk.embedding) != 768:
                            logger.warning(
                                f"Expected 768 dimensions for Gemini embedding, got {len(chunk.embedding)} for chunk {chunk.index}"
                            )
                        embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"

                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count,
                    )

                return document_id

    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("🧹 Clearing all existing data from databases...")
        logger.warning(
            "This will remove all documents, chunks, sessions, and knowledge graph data"
        )

        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                logger.info("Clearing PostgreSQL tables...")
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")

        logger.info("✅ Cleared PostgreSQL database")

        # Clean knowledge graph
        logger.info("Clearing knowledge graph...")
        await self.graph_builder.clear_graph()
        logger.info("✅ Cleared knowledge graph")

        logger.info("🎉 All data cleared successfully - starting fresh!")


async def main():
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector DB and knowledge graph using Gemini",
        epilog="""
Examples:
  python -m ingestion.ingest                    # Process first document only (default)
  python -m ingestion.ingest --all-docs         # Process all documents
  python -m ingestion.ingest --clear            # Clear all data and process first document
  python -m ingestion.ingest --clear --all-docs # Clear all data and process all documents
  python -m ingestion.ingest --clear-only       # Only clear all data (no ingestion)
  python -m ingestion.ingest --fast             # Skip knowledge graph building (faster)
  python -m ingestion.ingest --verbose          # Enable detailed logging
  python -m ingestion.ingest --dual-ingest      # Enable Onyx Cloud + Graphiti dual ingestion
  python -m ingestion.ingest --dual-ingest --all-docs # Dual ingest all documents
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--documents", "-d", default="documents", help="Documents folder path"
    )
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="Clean existing data before ingestion (deprecated, use --clear)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all existing data from both PostgreSQL and knowledge graph before starting fresh",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for splitting documents (optimized for Gemini)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap size (optimized for Gemini)",
    )
    parser.add_argument(
        "--no-semantic", action="store_true", help="Disable semantic chunking"
    )
    parser.add_argument(
        "--no-entities", action="store_true", help="Disable entity extraction"
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Fast mode: skip knowledge graph building",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="Process all documents (default: process only first document for testing)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--dual-ingest",
        action="store_true",
        help="Enable dual ingestion to Onyx Cloud before Graphiti processing (EPIC 3)",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear all existing data from both PostgreSQL and Neo4j/Graphiti without ingesting any documents",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting document ingestion with Gemini AI models")
    logger.info("Using Gemini embedding-001 (768 dimensions)")

    if not args.all_docs:
        logger.info(
            "🔥 SINGLE DOCUMENT MODE: Will process only the first document (use --all-docs to process all)"
        )
    else:
        logger.info("📚 FULL MODE: Will process all documents in the folder")

    # EPIC 3: Dual ingestion status
    if args.dual_ingest:
        logger.info(
            "🔮 DUAL INGESTION MODE: Documents will be ingested to Onyx Cloud before Graphiti processing"
        )
    else:
        logger.info(
            "📚 SINGLE INGESTION MODE: Documents will be processed locally only (use --dual-ingest for Onyx Cloud)"
        )

    # Handle clear/clean flags
    clean_before_ingest = args.clear or args.clean
    if args.clear:
        logger.info("🧹 CLEAR MODE: Will clear all existing data before ingestion")
    elif args.clean:
        logger.warning("⚠️  --clean flag is deprecated, use --clear instead")
        logger.info("🧹 CLEAN MODE: Will clean all existing data before ingestion")

    # Handle clear-only mode
    if args.clear_only:
        logger.info(
            "🧹 CLEAR-ONLY MODE: Will clear all existing data and exit (no ingestion)"
        )

        # Validate Gemini configuration (required for initialization)
        try:
            validate_gemini_configuration()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.error("Please ensure your .env file contains the correct API keys")
            return

        # Create minimal config for clearing only
        config = IngestionConfig(
            chunk_size=800,
            chunk_overlap=150,
            use_semantic_chunking=True,
            extract_entities=False,
            skip_graph_building=False,
            single_document_mode=True,
            enable_onyx_ingestion=False,
        )

        # Create pipeline for clearing only
        pipeline = DocumentIngestionPipeline(
            config=config,
            documents_folder=args.documents,
            clean_before_ingest=False,  # We'll call clean manually
        )

        try:
            # Initialize pipeline
            await pipeline.initialize()

            # Clear databases
            await pipeline._clean_databases()

            print("\n" + "=" * 50)
            print("🧹 CLEAR-ONLY COMPLETE")
            print("=" * 50)
            print("✅ PostgreSQL database cleared")
            print("✅ Neo4j/Graphiti knowledge graph cleared")
            print("✅ All existing data removed successfully")
            print("\n🚀 Databases are now clean and ready for fresh ingestion!")

        except Exception as e:
            logger.error(f"Failed to clear databases: {e}")
        finally:
            await pipeline.close()

        return  # Exit after clearing

    # Validate other flag combinations
    if args.clear_only and (args.clear or args.clean):
        logger.warning(
            "⚠️  --clear-only flag specified with --clear/--clean. Using --clear-only mode."
        )
        clean_before_ingest = False  # clear-only handles its own clearing

    # Debug environment loading
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    logger.debug(
        f"Project root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
    )

    # Early validation of configuration
    try:
        validate_gemini_configuration()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure your .env file contains the correct API keys")
        return

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=not args.no_entities,
        skip_graph_building=args.fast,
        single_document_mode=not args.all_docs,  # Default is single document mode unless --all-docs is specified
        enable_onyx_ingestion=args.dual_ingest,  # EPIC 3: Enable dual ingestion to Onyx Cloud
    )

    # Create and run pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=clean_before_ingest,
    )

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()

        # Validate Gemini configuration
        validate_gemini_configuration()

        results = await pipeline.ingest_documents(progress_callback)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Calculate Onyx statistics
        onyx_ingested_count = sum(1 for r in results if r.onyx_ingested)
        onyx_total_sections = sum(r.onyx_sections_count for r in results)

        # Print summary
        print("\n" + "=" * 50)
        print(
            "DUAL INGESTION SUMMARY (GEMINI + ONYX CLOUD)"
            if args.dual_ingest
            else "INGESTION SUMMARY (GEMINI-POWERED)"
        )
        print("=" * 50)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total entities extracted: {sum(r.entities_extracted for r in results)}")
        print(f"Total graph episodes: {sum(r.relationships_created for r in results)}")

        # EPIC 3: Onyx Cloud statistics
        if args.dual_ingest:
            print(
                f"Onyx Cloud ingested: {onyx_ingested_count}/{len(results)} documents"
            )
            print(f"Onyx total sections: {onyx_total_sections}")

        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("Embedding model: Gemini embedding-001 (768 dimensions)")
        print("LLM models: Gemini 2.0 Flash series")

        if args.dual_ingest:
            print("🔮 Dual ingestion: Local Graphiti + Onyx Cloud")
        else:
            print("📚 Single ingestion: Local Graphiti only")
        print()

        # Print individual results
        for result in results:
            status = "✓" if not result.errors else "✗"

            # Build result string
            result_parts = [
                f"{result.chunks_created} chunks",
                f"{result.entities_extracted} entities",
            ]

            # EPIC 3: Add Onyx status if dual ingestion was enabled
            if args.dual_ingest:
                onyx_status = "🔮" if result.onyx_ingested else "⚪"
                if result.onyx_ingested:
                    result_parts.append(
                        f"{onyx_status} Onyx: {result.onyx_sections_count} sections"
                    )
                else:
                    result_parts.append(f"{onyx_status} Onyx: failed")

            print(f"{status} {result.title}: {', '.join(result_parts)}")

            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")

    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
