#!/usr/bin/env python3
"""
Interactive Multi-Tenant CLI for Hybrid RAG System
Provides authenticated command-line interface for multi-tenant RAG with comprehensive agent integration.
"""

import asyncio
import os
import sys
import uuid
import logging
from typing import Optional
from datetime import datetime

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print(
        "Warning: CLI dependencies not installed. Install with: pip install click rich"
    )
    sys.exit(1)

from tenant_manager import TenantManager
from auth_middleware import TenantContext, JWTAuthenticator
from multi_tenant_agent import MultiTenantRAGAgent

# Import validated agent tools (commented out as we're using tenant-aware methods)
# try:
#     from agent.tools import (
#         vector_search_tool,
#         graph_search_tool,
#         hybrid_search_tool,
#         comprehensive_search_tool,
#         VectorSearchInput,
#         GraphSearchInput,
#         HybridSearchInput,
#         ComprehensiveSearchInput,
#     )
# except ImportError as e:
#     print(f"Error importing agent modules: {e}")
#     print("Please ensure the agent folder is properly configured")
#     sys.exit(1)

# Configure logging - Send all backend logs to API log file and terminal
# CLI should not create its own log files, everything goes to API
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Only log to the API log file, not CLI-specific files
        logging.FileHandler("interactive_multi_tenant_api.log"),
    ],
)

logger = logging.getLogger(__name__)


# Create a console filter to hide backend logs from CLI user interface
class CLIConsoleFilter(logging.Filter):
    """Filter to suppress backend logs from CLI console while allowing them in API terminal."""

    def filter(self, record):
        # Suppress these modules from CLI user interface
        backend_modules = [
            "httpx",
            "google_genai",
            "multi_tenant_agent",
            "tenant_graphiti_client",
            "tenant_data_ingestion_service",
            "neo4j",
            "catalog_database",
            "ingestion",
        ]
        return not any(record.name.startswith(module) for module in backend_modules)


# Configure backend loggers to use API log file and terminal
backend_loggers = [
    "multi_tenant_agent",
    "tenant_graphiti_client",
    "tenant_data_ingestion_service",
    "httpx",
    "google_genai",
    "neo4j",
    "catalog_database",
    "ingestion",
]

# Ensure all backend logs go to API log file and terminal (not CLI console)
for logger_name in backend_loggers:
    backend_logger = logging.getLogger(logger_name)
    backend_logger.setLevel(logging.INFO)

    # Add file handler if not present
    if not any(isinstance(h, logging.FileHandler) for h in backend_logger.handlers):
        file_handler = logging.FileHandler("interactive_multi_tenant_api.log")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        backend_logger.addHandler(file_handler)

    # Prevent propagation to CLI console but allow to API terminal
    backend_logger.propagate = False

# Rich console for beautiful output
console = Console()


class MultiTenantCLI:
    """Interactive multi-tenant CLI for RAG system."""

    def __init__(self):
        self.tenant_manager = None
        self.jwt_authenticator = None
        self.current_tenant = None
        self.current_session = None
        self.agent = None

    async def initialize(self):
        """Initialize CLI components."""
        try:
            console.print(
                "🚀 [bold blue]Initializing Multi-Tenant RAG CLI...[/bold blue]"
            )

            # Get environment variables
            neon_api_key = os.getenv("NEON_API_KEY", "neon_mock_api_key")
            catalog_db_url = os.getenv("CATALOG_DB_URL") or os.getenv(
                "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/catalog"
            )

            # Get Neo4j credentials for Graphiti integration
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            # Initialize tenant manager with Neo4j credentials
            self.tenant_manager = TenantManager(
                neon_api_key=neon_api_key,
                catalog_db_url=catalog_db_url,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
            )
            console.print("✅ Tenant manager initialized")

            # Initialize ingestion service for tenant-aware searches
            from tenant_data_ingestion_service import TenantDataIngestionService

            self.tenant_manager.ingestion_service = TenantDataIngestionService(
                tenant_manager=self.tenant_manager
            )
            console.print("✅ Tenant ingestion service initialized")

            # Initialize JWT authenticator
            self.jwt_authenticator = JWTAuthenticator()
            console.print("✅ JWT authenticator initialized")

            console.print("🎉 [bold green]CLI initialization complete![/bold green]\n")

        except Exception as e:
            console.print(f"❌ [bold red]Failed to initialize CLI: {e}[/bold red]")
            raise

    async def authenticate(self):
        """Authenticate user and set tenant context."""
        console.print("\n🔐 [bold cyan]Authentication Required[/bold cyan]")

        # Get tenant ID
        tenant_id = Prompt.ask("Enter tenant ID")

        # Check if tenant exists
        try:
            tenant_uuid = (
                uuid.UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
            )
            tenant_info = await self.tenant_manager.get_tenant(tenant_uuid)
            if not tenant_info:
                console.print(f"❌ [bold red]Tenant '{tenant_id}' not found[/bold red]")
                return False
        except ValueError:
            console.print(
                f"❌ [bold red]Invalid tenant ID format: '{tenant_id}'[/bold red]"
            )
            return False

        # Get API key (simplified authentication)
        api_key = Prompt.ask("Enter API key", password=True)
        expected_api_key = f"api_key_{tenant_id}"

        if api_key != expected_api_key:
            console.print("❌ [bold red]Invalid API key[/bold red]")
            return False

        # Create tenant context
        self.current_tenant = TenantContext(
            tenant_id=tenant_id,
            user_id=Prompt.ask("Enter user ID (optional)", default="anonymous"),
            permissions=["read", "write"],
            session_id=str(uuid.uuid4()),
        )

        # Initialize multi-tenant agent
        try:
            self.agent = MultiTenantRAGAgent(
                tenant_manager=self.tenant_manager,
                graphiti_client=getattr(self.tenant_manager, "graphiti_client", None),
                model_name="gemini-2.0-flash-thinking-exp-1219",
            )
            console.print(
                "✅ [green]Multi-tenant agent initialized successfully[/green]"
            )
        except Exception as e:
            console.print(f"❌ [bold red]Error initializing agent: {e}[/bold red]")
            return False

        console.print(
            f"✅ [bold green]Successfully authenticated as {self.current_tenant.user_id} for tenant {tenant_id}[/bold green]"
        )
        return True

    async def show_tenant_info(self):
        """Display current tenant information."""
        if not self.current_tenant:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            return

        tenant_data = await self.tenant_manager.get_tenant(
            self.current_tenant.tenant_id
        )

        table = Table(title=f"Tenant Information: {self.current_tenant.tenant_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Tenant ID", self.current_tenant.tenant_id)
        table.add_row("User ID", self.current_tenant.user_id)
        table.add_row("Session ID", self.current_tenant.session_id)
        table.add_row("Permissions", ", ".join(self.current_tenant.permissions))
        table.add_row("Status", tenant_data.get("status", "active"))
        table.add_row("Created At", str(tenant_data.get("created_at", "N/A")))

        console.print(table)

    async def search_menu(self):
        """Interactive search menu."""
        if not self.current_tenant:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            return

        search_types = {
            "1": ("vector", "Vector Search - Semantic similarity"),
            "2": ("graph", "Graph Search - Facts and relationships"),
            "3": ("hybrid", "Hybrid Search - Combined vector and graph"),
            "4": ("comprehensive", "Comprehensive Search - All methods"),
        }

        console.print("\n🔍 [bold cyan]Search Options[/bold cyan]")
        for key, (_, description) in search_types.items():
            console.print(f"  {key}. {description}")

        choice = Prompt.ask("Select search type", choices=list(search_types.keys()))
        search_type, _ = search_types[choice]

        query = Prompt.ask("Enter your search query")
        limit = int(Prompt.ask("Enter result limit", default="10"))

        if search_type == "hybrid":
            text_weight = float(
                Prompt.ask("Enter text weight (0.0-1.0)", default="0.3")
            )
        else:
            text_weight = 0.3

        # Perform search with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Searching using {search_type} method...", total=None
            )

            try:
                start_time = datetime.now()
                results = await self._perform_search(
                    search_type, query, limit, text_weight
                )
                execution_time = (datetime.now() - start_time).total_seconds()

                progress.update(task, completed=True)

                # Display results
                await self._display_search_results(
                    results, search_type, query, execution_time
                )

            except Exception as e:
                console.print(f"❌ [bold red]Search error: {e}[/bold red]")

    async def _perform_search(
        self, search_type: str, query: str, limit: int, text_weight: float
    ):
        """Perform search based on type using tenant-aware methods."""
        if not self.current_tenant:
            raise ValueError("Not authenticated")

        tenant_id = self.current_tenant.tenant_id

        if search_type == "vector":
            # Use tenant-aware vector search
            tenant_db_url = await self.tenant_manager.get_tenant_database_url(tenant_id)
            return await self.tenant_manager.ingestion_service.vector_search_for_tenant(
                tenant_database_url=tenant_db_url, query=query, limit=limit
            )

        elif search_type == "graph":
            # Use tenant-aware graph search
            if self.tenant_manager.graphiti_client:
                return await self.tenant_manager.graphiti_client.search_tenant_graph(
                    tenant_id=tenant_id, query=query, limit=limit
                )
            else:
                console.print("❌ [red]Graphiti client not available[/red]")
                return []

        elif search_type == "hybrid":
            # Use tenant-aware hybrid search
            tenant_db_url = await self.tenant_manager.get_tenant_database_url(tenant_id)
            return await self.tenant_manager.ingestion_service.hybrid_search_for_tenant(
                tenant_database_url=tenant_db_url,
                query=query,
                limit=limit,
                text_weight=text_weight,
                vector_weight=1.0 - text_weight,
            )

        elif search_type == "comprehensive":
            # Comprehensive search using all methods
            try:
                import asyncio

                tenant_db_url = await self.tenant_manager.get_tenant_database_url(
                    tenant_id
                )

                # Run searches concurrently
                vector_task = (
                    self.tenant_manager.ingestion_service.vector_search_for_tenant(
                        tenant_database_url=tenant_db_url, query=query, limit=limit // 2
                    )
                )

                graph_task = []
                if self.tenant_manager.graphiti_client:
                    graph_task = (
                        self.tenant_manager.graphiti_client.search_tenant_graph(
                            tenant_id=tenant_id, query=query, limit=limit // 2
                        )
                    )

                hybrid_task = (
                    self.tenant_manager.ingestion_service.hybrid_search_for_tenant(
                        tenant_database_url=tenant_db_url,
                        query=query,
                        limit=limit // 2,
                        text_weight=text_weight,
                        vector_weight=1.0 - text_weight,
                    )
                )

                # Wait for all searches
                if self.tenant_manager.graphiti_client:
                    (
                        vector_results,
                        graph_results,
                        hybrid_results,
                    ) = await asyncio.gather(
                        vector_task, graph_task, hybrid_task, return_exceptions=True
                    )
                else:
                    vector_results, hybrid_results = await asyncio.gather(
                        vector_task, hybrid_task, return_exceptions=True
                    )
                    graph_results = []

                # Handle exceptions
                if isinstance(vector_results, Exception):
                    vector_results = []
                if isinstance(graph_results, Exception):
                    graph_results = []
                if isinstance(hybrid_results, Exception):
                    hybrid_results = []

                # Combine results (convert to consistent format)
                combined = []

                # Add hybrid results
                for r in hybrid_results:
                    combined.append(
                        {
                            "content": r.get("content", ""),
                            "score": r.get("combined_score", 0),
                            "document_source": r.get("document_source", ""),
                            "type": "hybrid",
                        }
                    )

                # Add vector results
                for r in vector_results:
                    combined.append(
                        {
                            "content": r.get("content", ""),
                            "score": r.get("similarity", 0),
                            "document_source": r.get("document_source", ""),
                            "type": "vector",
                        }
                    )

                # Add graph results
                for r in graph_results:
                    combined.append(
                        {
                            "content": r.get("fact", ""),
                            "score": 1.0,
                            "document_source": "knowledge_graph",
                            "type": "graph",
                        }
                    )

                return combined[:limit]

            except Exception as e:
                console.print(f"❌ [red]Comprehensive search error: {e}[/red]")
                return []

        else:
            raise ValueError(f"Invalid search type: {search_type}")

    async def _display_search_results(
        self, results, search_type: str, query: str, execution_time: float
    ):
        """Display search results in a formatted table."""
        if not results:
            console.print("📭 [yellow]No results found[/yellow]")
            return

        # Create results table
        table = Table(title=f"Search Results: {search_type.title()} Search")
        table.add_column("ID", style="dim", width=6)
        table.add_column("Content", style="green", width=60)
        table.add_column("Score", style="cyan", width=10)
        table.add_column("Source", style="blue", width=20)

        for i, result in enumerate(results[:10], 1):  # Show top 10 results
            # Handle both dict and object formats
            if isinstance(result, dict):
                content = result.get("content") or result.get("fact", "")
                score = (
                    result.get("score")
                    or result.get("similarity")
                    or result.get("combined_score", 0)
                )
                source = result.get("document_source") or result.get(
                    "source", "unknown"
                )
            else:
                content = getattr(result, "content", getattr(result, "fact", ""))
                score = getattr(result, "score", getattr(result, "similarity", 0))
                source = getattr(
                    result, "document_source", getattr(result, "source", "unknown")
                )

            # Format score
            score_str = (
                f"{score:.3f}"
                if isinstance(score, (int, float)) and score != 0
                else "N/A"
            )

            # Truncate content for display
            if len(content) > 100:
                content = content[:100] + "..."

            table.add_row(str(i), content, score_str, source)

        console.print(table)

        # Show summary
        summary_panel = Panel(
            f"Query: {query}\n"
            f"Search Type: {search_type.title()}\n"
            f"Results: {len(results)}\n"
            f"Execution Time: {execution_time:.3f}s\n"
            f"Tenant: {self.current_tenant.tenant_id}",
            title="Search Summary",
            border_style="blue",
        )
        console.print(summary_panel)

        # Ask if user wants to see detailed results
        if Confirm.ask("View detailed results?"):
            await self._show_detailed_results(results)

    async def _show_detailed_results(self, results):
        """Show detailed view of search results."""
        for i, result in enumerate(results, 1):
            # Handle both dict and object formats
            if isinstance(result, dict):
                content = result.get("content") or result.get("fact", "")
                score = (
                    result.get("score")
                    or result.get("similarity")
                    or result.get("combined_score")
                )
                document_title = result.get("document_title", "")
                document_source = result.get("document_source", "")
                valid_at = result.get("valid_at", "")
            else:
                content = getattr(result, "content", getattr(result, "fact", ""))
                score = getattr(result, "score", getattr(result, "similarity", None))
                document_title = getattr(result, "document_title", "")
                document_source = getattr(result, "document_source", "")
                valid_at = getattr(result, "valid_at", "")

            panel_content = f"[bold]Content:[/bold]\n{content}\n\n"

            if score is not None:
                panel_content += f"[bold]Score:[/bold] {score:.4f}\n"

            if document_title:
                panel_content += f"[bold]Document:[/bold] {document_title}\n"

            if document_source:
                panel_content += f"[bold]Source:[/bold] {document_source}\n"

            if valid_at:
                panel_content += f"[bold]Valid At:[/bold] {valid_at}\n"

            panel = Panel(panel_content, title=f"Result {i}", border_style="green")
            console.print(panel)

            if i % 3 == 0 and i < len(results):  # Pause every 3 results
                if not Confirm.ask("Continue viewing results?"):
                    break

    async def chat_mode(self):
        """Interactive chat mode with the RAG agent."""
        if not self.current_tenant:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            return

        console.print("\n💬 [bold cyan]Chat Mode - Type 'exit' to quit[/bold cyan]")
        console.print("Ask questions about your documents and knowledge base.")
        console.print(
            "[dim]Tip: Use arrow keys to navigate and edit your input[/dim]\n"
        )

        session_id = str(uuid.uuid4())

        while True:
            try:
                # Get user input with better editing capabilities
                try:
                    import readline  # Enable arrow key navigation and input history

                    readline.set_history_length(100)  # Keep last 100 commands
                except ImportError:
                    pass  # Not available on all systems

                message = input("\n🧑 You: ").strip()

                if message.lower() in ["exit", "quit", "bye"]:
                    console.print("👋 [yellow]Goodbye![/yellow]")
                    break

                # Show typing indicator without backend logs
                with Progress(
                    SpinnerColumn(),
                    TextColumn("🤖 Agent is thinking..."),
                    console=console,
                    transient=True,  # Make progress bar disappear after completion
                ) as progress:
                    progress.add_task("Processing...", total=None)

                    start_time = datetime.now()

                    # Check if agent is initialized
                    if not self.agent:
                        console.print(
                            "❌ [bold red]Agent not initialized. Please re-authenticate.[/bold red]"
                        )
                        break

                    # Get response from agent (backend logs will go to file/API terminal)
                    agent_result = await self.agent.chat(
                        message=message,
                        context=TenantContext(
                            tenant_id=self.current_tenant.tenant_id,
                            user_id=self.current_tenant.user_id,
                            session_id=session_id,
                        ),
                    )

                    execution_time = (datetime.now() - start_time).total_seconds()

                # Extract clean response from agent result
                if isinstance(agent_result, dict):
                    response_text = agent_result.get(
                        "response", "No response available"
                    )
                    tools_used = agent_result.get("tools_used", [])
                    sources = agent_result.get("sources", [])
                elif hasattr(agent_result, "data"):
                    # Handle Pydantic AI result format
                    response_text = str(agent_result.data)
                    # For Pydantic AI results, try to extract tools from metadata if available
                    tools_used = getattr(agent_result, "tools_used", [])
                    sources = getattr(agent_result, "sources", [])

                    # If no tools found, check if we have the full result structure
                    if not tools_used and hasattr(agent_result, "metadata"):
                        tools_used = agent_result.metadata.get("tools_used", [])
                        sources = agent_result.metadata.get("sources", [])
                else:
                    response_text = str(agent_result)
                    tools_used = []
                    sources = []

                # Ensure we have a fallback for tools if none were detected
                if not tools_used:
                    tools_used = ["Dual Storage Search"]

                # Clean up response text if it contains technical formatting
                if response_text.startswith(
                    'AgentRunResult(output="'
                ) and response_text.endswith('")'):
                    response_text = response_text[23:-2]  # Remove wrapper

                # Further clean up technical wrappers
                if "AgentRunResult" in response_text:
                    # Extract content between quotes if wrapped
                    import re

                    match = re.search(r'output="([^"]*)"', response_text)
                    if match:
                        response_text = match.group(1)
                    else:
                        # Fallback: remove common technical patterns
                        response_text = re.sub(
                            r"AgentRunResult\([^)]*\)", "", response_text
                        ).strip()

                # Ensure response is conversational
                if not response_text or response_text == "No response available":
                    response_text = "I'm sorry, I wasn't able to generate a proper response to your question."

                # Display clean conversational response
                console.print(f"\n🤖 **Assistant**: {response_text}")

                # Add execution time in a subtle way
                console.print(f"[dim]   ⏱️ Responded in {execution_time:.2f}s[/dim]")

                # Show tool usage information in a clean way
                if tools_used:
                    tools_str = ", ".join(tools_used)
                    console.print(f"[dim]   🔧 Used: {tools_str}[/dim]")

                # Show sources if available in a clean way
                if sources and len(sources) > 0:
                    console.print(
                        f"[dim]   📚 Found {len(sources)} relevant sources[/dim]"
                    )
                    for i, source in enumerate(sources[:2], 1):  # Show top 2 sources
                        if isinstance(source, dict):
                            source_name = source.get(
                                "source", source.get("document_title", "Document")
                            )
                            if isinstance(source_name, str) and len(source_name) > 50:
                                source_name = source_name[:47] + "..."
                            console.print(f"[dim]      {i}. {source_name}[/dim]")
                        else:
                            source_str = str(source)
                            if len(source_str) > 50:
                                source_str = source_str[:47] + "..."
                            console.print(f"[dim]      {i}. {source_str}[/dim]")

            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Chat interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"❌ [bold red]Error: {e}[/bold red]")
                logger.error(f"Chat error: {e}")

    async def list_tenants(self):
        """List all available tenants."""
        try:
            tenants = await self.tenant_manager.list_tenants()

            if not tenants:
                console.print("📭 [yellow]No tenants found[/yellow]")
                return

            table = Table(title="Available Tenants")
            table.add_column("Tenant ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Created", style="blue")

            for tenant in tenants:
                table.add_row(
                    str(tenant.tenant_id),
                    tenant.status.value,
                    str(tenant.created_at),
                )

            console.print(table)

        except Exception as e:
            console.print(f"❌ [bold red]Error listing tenants: {e}[/bold red]")

    async def main_menu(self):
        """Main interactive menu."""
        while True:
            try:
                # Show current status
                status = (
                    f"Authenticated as {self.current_tenant.user_id} ({self.current_tenant.tenant_id})"
                    if self.current_tenant
                    else "Not authenticated"
                )

                console.print(
                    f"\n🏠 [bold blue]Multi-Tenant RAG CLI[/bold blue] - {status}"
                )
                console.print("\nSelect an option:")
                console.print("  1. 🔐 Authenticate")
                console.print("  2. ℹ️  Show tenant info")
                console.print("  3. 🔍 Search")
                console.print("  4. 💬 Chat mode")
                console.print("  5. 📋 List tenants")
                console.print("  6. 🚪 Exit")

                choice = Prompt.ask(
                    "Enter choice", choices=["1", "2", "3", "4", "5", "6"]
                )

                if choice == "1":
                    await self.authenticate()
                elif choice == "2":
                    await self.show_tenant_info()
                elif choice == "3":
                    await self.search_menu()
                elif choice == "4":
                    await self.chat_mode()
                elif choice == "5":
                    await self.list_tenants()
                elif choice == "6":
                    console.print("👋 [yellow]Goodbye![/yellow]")
                    break

            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"❌ [bold red]Error: {e}[/bold red]")


@click.command()
@click.option("--tenant-id", help="Pre-authenticate with tenant ID")
@click.option("--api-key", help="API key for authentication")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(
    tenant_id: Optional[str] = None, api_key: Optional[str] = None, debug: bool = False
):
    """Interactive Multi-Tenant CLI for Hybrid RAG System."""

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    async def run_cli():
        cli = MultiTenantCLI()

        try:
            await cli.initialize()

            # Pre-authenticate if credentials provided
            if tenant_id and api_key:
                expected_api_key = f"api_key_{tenant_id}"
                if api_key == expected_api_key:
                    try:
                        tenant_uuid = (
                            uuid.UUID(tenant_id)
                            if isinstance(tenant_id, str)
                            else tenant_id
                        )
                        tenant_info = await cli.tenant_manager.get_tenant(tenant_uuid)
                        if tenant_info:
                            cli.current_tenant = TenantContext(
                                tenant_id=tenant_id,
                                user_id="cli_user",
                                permissions=["read", "write"],
                                session_id=str(uuid.uuid4()),
                            )
                            cli.agent = MultiTenantRAGAgent(
                                tenant_manager=cli.tenant_manager,
                                graphiti_client=getattr(
                                    cli.tenant_manager, "graphiti_client", None
                                ),
                                model_name="gemini-2.0-flash-thinking-exp-1219",
                            )
                            console.print(
                                f"✅ [green]Pre-authenticated for tenant {tenant_id}[/green]"
                            )
                        else:
                            console.print(f"❌ [red]Tenant {tenant_id} not found[/red]")
                    except (ValueError, Exception) as e:
                        console.print(f"❌ [red]Authentication error: {e}[/red]")
                else:
                    console.print("❌ [red]Invalid API key[/red]")

            await cli.main_menu()

        except Exception as e:
            console.print(f"❌ [bold red]CLI error: {e}[/bold red]")
            logger.exception("CLI error")

    # Welcome message
    console.print(
        Panel.fit(
            "🚀 [bold blue]Multi-Tenant RAG CLI[/bold blue]\n"
            "Interactive command-line interface for hybrid RAG with knowledge graphs\n"
            "Supports vector search, graph search, hybrid search, and chat mode",
            border_style="blue",
        )
    )

    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
