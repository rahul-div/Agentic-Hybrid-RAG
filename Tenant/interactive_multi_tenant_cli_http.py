#!/usr/bin/env python3
"""
HTTP-Based Interactive Multi-Tenant CLI for Hybrid RAG System
Provides authenticated command-line interface using HTTP API endpoints with JWT authentication.
"""

import asyncio
import os
import sys
import uuid
import logging
from typing import Optional, Dict, Any, List
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
    import httpx
except ImportError:
    print(
        "Warning: CLI dependencies not installed. Install with: pip install click rich httpx"
    )
    sys.exit(1)

# Configure logging - CLI logs only for errors, all backend activity goes to API
logging.basicConfig(
    level=logging.ERROR,  # Only show errors in CLI
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


class HTTPMultiTenantCLI:
    """HTTP-based multi-tenant CLI for RAG system using API endpoints."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")
        self.jwt_token = None
        self.current_tenant_id = None
        self.current_user_id = None
        # Regular client for API calls
        self.client = httpx.AsyncClient(timeout=30.0)
        # Extended timeout client for file uploads
        self.upload_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for uploads

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
        await self.upload_client.aclose()

    async def check_api_health(self) -> bool:
        """Check if API server is running and healthy."""
        try:
            response = await self.client.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                console.print(
                    f"✅ API Server: {health_data['status']} (v{health_data['version']})"
                )
                console.print(f"   📊 Active tenants: {health_data['tenant_count']}")
                return True
            else:
                console.print(f"❌ API Server unhealthy: {response.status_code}")
                return False
        except httpx.RequestError as e:
            console.print(
                f"❌ Cannot connect to API server at {self.api_base_url}: {e}"
            )
            return False

    async def authenticate(self) -> bool:
        """Authenticate user via API endpoint and get JWT token."""
        console.print("\n🔐 [bold cyan]Authentication Required[/bold cyan]")

        # Get credentials
        tenant_id = Prompt.ask("Enter tenant ID")
        api_key = Prompt.ask("Enter API key", password=True)
        user_id = Prompt.ask("Enter user ID (optional)", default="cli_user")

        try:
            # Call authentication endpoint
            auth_data = {"tenant_id": tenant_id, "api_key": api_key, "user_id": user_id}

            response = await self.client.post(
                f"{self.api_base_url}/auth/login", json=auth_data
            )

            if response.status_code == 200:
                auth_result = response.json()
                self.jwt_token = auth_result["access_token"]
                self.current_tenant_id = tenant_id
                self.current_user_id = user_id

                # Update client headers with token
                auth_header = {"Authorization": f"Bearer {self.jwt_token}"}
                self.client.headers.update(auth_header)
                self.upload_client.headers.update(auth_header)

                console.print(
                    f"✅ [bold green]Successfully authenticated as {user_id} for tenant {tenant_id}[/bold green]"
                )
                console.print(
                    f"[dim]Token expires: {auth_result.get('expires_at', 'N/A')}[/dim]"
                )
                return True
            else:
                error_detail = response.json().get("detail", "Authentication failed")
                console.print(f"❌ [bold red]{error_detail}[/bold red]")
                return False

        except httpx.RequestError as e:
            console.print(f"❌ [bold red]Network error: {e}[/bold red]")
            return False
        except Exception as e:
            console.print(f"❌ [bold red]Authentication error: {e}[/bold red]")
            return False

    async def show_tenant_info(self):
        """Display current tenant information via API."""
        if not self.jwt_token:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            return

        try:
            response = await self.client.get(f"{self.api_base_url}/tenants/info")

            if response.status_code == 200:
                tenant_data = response.json()

                table = Table(title=f"Tenant Information: {self.current_tenant_id}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Tenant ID", self.current_tenant_id)
                table.add_row("User ID", self.current_user_id)
                table.add_row("Status", tenant_data.get("status", "active"))
                table.add_row("Created At", str(tenant_data.get("created_at", "N/A")))

                # Add metadata if available
                metadata = tenant_data.get("metadata", {})
                for key, value in metadata.items():
                    table.add_row(f"Metadata: {key}", str(value))

                console.print(table)
            else:
                error_detail = response.json().get(
                    "detail", "Failed to get tenant info"
                )
                console.print(f"❌ [bold red]{error_detail}[/bold red]")

        except httpx.RequestError as e:
            console.print(f"❌ [bold red]Network error: {e}[/bold red]")
        except Exception as e:
            console.print(f"❌ [bold red]Error: {e}[/bold red]")

    async def search_menu(self):
        """Interactive search menu using API endpoints."""
        if not self.jwt_token:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            return

        search_types = {
            "1": ("vector", "Vector Search - Semantic similarity"),
            "2": ("graph", "Graph Search - Facts and relationships"),
            "3": ("hybrid", "Hybrid Search - Vector similarity + keyword matching"),
            "4": ("comprehensive", "Comprehensive Search - Vector + Graph + Hybrid"),
        }

        console.print("\n🔍 [bold cyan]Search Options[/bold cyan]")
        for key, (_, description) in search_types.items():
            console.print(f"  {key}. {description}")

        choice = Prompt.ask("Select search type", choices=list(search_types.keys()))
        search_type, _ = search_types[choice]

        query = Prompt.ask("Enter your search query")
        limit = int(Prompt.ask("Enter result limit", default="10"))

        text_weight = 0.3
        if search_type == "hybrid":
            text_weight = float(
                Prompt.ask("Enter text weight (0.0-1.0)", default="0.3")
            )

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
                results = await self._perform_search_api(
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

    async def _perform_search_api(
        self, search_type: str, query: str, limit: int, text_weight: float
    ) -> Dict[str, Any]:
        """Perform search using API endpoint."""
        search_data = {
            "query": query,
            "search_type": search_type,
            "limit": limit,
            "text_weight": text_weight,
        }

        try:
            response = await self.client.post(
                f"{self.api_base_url}/search", json=search_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Search failed")
                raise Exception(f"Search API error: {error_detail}")

        except httpx.RequestError as e:
            raise Exception(f"Network error: {e}")

    async def _display_search_results(
        self,
        search_response: Dict[str, Any],
        search_type: str,
        query: str,
        execution_time: float,
    ):
        """Display search results from API response."""
        results = search_response.get("results", [])

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
            content = result.get("content", "")
            score = result.get("score")
            source = result.get("source", "unknown")

            # Format score
            score_str = (
                f"{score:.3f}"
                if isinstance(score, (int, float)) and score is not None
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
            f"Results: {search_response.get('total_results', len(results))}\n"
            f"API Execution Time: {search_response.get('execution_time', 0):.3f}s\n"
            f"Total Time: {execution_time:.3f}s\n"
            f"Tenant: {self.current_tenant_id}",
            title="Search Summary",
            border_style="blue",
        )
        console.print(summary_panel)

        # Ask if user wants to see detailed results
        if Confirm.ask("View detailed results?"):
            await self._show_detailed_results(results)

    async def _show_detailed_results(self, results: List[Dict[str, Any]]):
        """Show detailed view of search results."""
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            score = result.get("score")
            source = result.get("source", "")
            metadata = result.get("metadata", {})

            panel_content = f"[bold]Content:[/bold]\n{content}\n\n"

            if score is not None:
                panel_content += f"[bold]Score:[/bold] {score:.4f}\n"

            if source:
                panel_content += f"[bold]Source:[/bold] {source}\n"

            # Add metadata information
            if metadata:
                panel_content += "[bold]Metadata:[/bold]\n"
                for key, value in metadata.items():
                    panel_content += f"  {key}: {value}\n"

            panel = Panel(panel_content, title=f"Result {i}", border_style="green")
            console.print(panel)

            if i % 3 == 0 and i < len(results):  # Pause every 3 results
                if not Confirm.ask("Continue viewing results?"):
                    break

    async def chat_mode(self):
        """Interactive chat mode using API endpoint."""
        if not self.jwt_token:
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

                # Show typing indicator
                with Progress(
                    SpinnerColumn(),
                    TextColumn("🤖 Agent is thinking..."),
                    console=console,
                    transient=True,  # Make progress bar disappear after completion
                ) as progress:
                    progress.add_task("Processing...", total=None)

                    start_time = datetime.now()

                    # Send chat request to API
                    chat_data = {"message": message, "session_id": session_id}

                    try:
                        response = await self.client.post(
                            f"{self.api_base_url}/chat", json=chat_data
                        )

                        execution_time = (datetime.now() - start_time).total_seconds()

                        if response.status_code == 200:
                            chat_result = response.json()

                            response_text = chat_result.get(
                                "response", "No response available"
                            )
                            sources = chat_result.get("sources", [])
                            tools_used = chat_result.get("tools_used", [])
                            api_execution_time = chat_result.get("execution_time", 0)

                            # Display clean conversational response
                            console.print(f"\n🤖 **Assistant**: {response_text}")

                            # Add execution time information
                            console.print(
                                f"[dim]   ⏱️ API: {api_execution_time:.2f}s, Total: {execution_time:.2f}s[/dim]"
                            )

                            # Show tools used if available
                            if tools_used and len(tools_used) > 0:
                                tools_str = ", ".join(tools_used)
                                console.print(
                                    f"[dim]   🔧 Tools used: {tools_str}[/dim]"
                                )

                            # Show sources if available
                            if sources and len(sources) > 0:
                                console.print(
                                    f"[dim]   📚 Found {len(sources)} relevant sources[/dim]"
                                )
                                for i, source in enumerate(
                                    sources[:2], 1
                                ):  # Show top 2 sources
                                    source_name = source.get(
                                        "source",
                                        source.get("document_title", "Document"),
                                    )
                                    if (
                                        isinstance(source_name, str)
                                        and len(source_name) > 50
                                    ):
                                        source_name = source_name[:47] + "..."
                                    console.print(
                                        f"[dim]      {i}. {source_name}[/dim]"
                                    )

                        else:
                            error_detail = response.json().get(
                                "detail", "Chat request failed"
                            )
                            console.print(
                                f"❌ [bold red]Error: {error_detail}[/bold red]"
                            )

                    except httpx.RequestError as e:
                        console.print(f"❌ [bold red]Network error: {e}[/bold red]")
                    except Exception as e:
                        console.print(f"❌ [bold red]Chat error: {e}[/bold red]")

            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Chat interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"❌ [bold red]Error: {e}[/bold red]")
                logger.error(f"Chat error: {e}")

    async def list_tenants(self):
        """List all available tenants (requires admin privileges or specific endpoint)."""
        try:
            # Note: This might require admin privileges in a real system
            # For now, we'll just show current tenant info
            console.print(
                "📋 [yellow]Tenant listing not available through current API endpoints[/yellow]"
            )
            console.print("Current tenant information:")
            await self.show_tenant_info()

        except Exception as e:
            console.print(f"❌ [bold red]Error: {e}[/bold red]")

    async def create_tenant(self):
        """Create a new tenant via API (no authentication required)."""
        console.print("\n➕ [bold cyan]Create New Tenant[/bold cyan]")

        try:
            # Get tenant details
            tenant_name = Prompt.ask("Enter tenant name")
            tenant_email = Prompt.ask("Enter tenant email")
            region = Prompt.ask("Enter region", default="aws-us-east-1")
            plan = Prompt.ask("Enter plan", default="basic")

            # Prepare tenant creation request
            tenant_data = {
                "name": tenant_name,
                "email": tenant_email,
                "region": region,
                "plan": plan,
            }

            # Show progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating tenant...", total=None)

                try:
                    response = await self.client.post(
                        f"{self.api_base_url}/tenants", json=tenant_data
                    )

                    progress.update(task, completed=True)

                    if response.status_code == 200:
                        tenant_result = response.json()

                        console.print(
                            "✅ [bold green]Tenant created successfully![/bold green]"
                        )

                        # Display tenant information
                        table = Table(title="New Tenant Information")
                        table.add_column("Property", style="cyan")
                        table.add_column("Value", style="green")

                        table.add_row("Tenant ID", tenant_result["tenant_id"])
                        table.add_row("Name", tenant_result["tenant_name"])
                        table.add_row("Email", tenant_result["tenant_email"])
                        table.add_row("Status", tenant_result["status"])
                        table.add_row("Region", tenant_result["region"])
                        table.add_row("Plan", tenant_result["plan"])
                        table.add_row("Created At", str(tenant_result["created_at"]))

                        if tenant_result.get("neon_project_id"):
                            table.add_row(
                                "Neon Project ID", tenant_result["neon_project_id"]
                            )

                        console.print(table)

                        # Show authentication info
                        api_key = f"api_key_{tenant_result['tenant_id']}"
                        console.print(
                            f"\n🔑 [bold yellow]Your API key:[/bold yellow] {api_key}"
                        )
                        console.print(
                            "[dim]Save this API key - you'll need it to authenticate![/dim]"
                        )

                    else:
                        error_detail = response.json().get(
                            "detail", "Tenant creation failed"
                        )
                        console.print(f"❌ [bold red]{error_detail}[/bold red]")

                except httpx.RequestError as e:
                    console.print(f"❌ [bold red]Network error: {e}[/bold red]")
                except Exception as e:
                    console.print(f"❌ [bold red]Tenant creation error: {e}[/bold red]")

        except Exception as e:
            console.print(f"❌ [bold red]Error: {e}[/bold red]")

    async def upload_document(self):
        """Upload a document for the authenticated tenant with ingestion options."""
        if not self.jwt_token:
            console.print("❌ [bold red]Not authenticated[/bold red]")
            console.print("[dim]Please authenticate first using option 1[/dim]")
            return

        console.print("\n📄 [bold cyan]Upload Document[/bold cyan]")

        try:
            # Get file path
            file_path = Prompt.ask("Enter file path")

            if not os.path.exists(file_path):
                console.print(f"❌ [bold red]File not found: {file_path}[/bold red]")
                return

            if not os.path.isfile(file_path):
                console.print(
                    f"❌ [bold red]Path is not a file: {file_path}[/bold red]"
                )
                return

            # Get file info
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            console.print(f"📁 File: {filename}")
            console.print(f"📏 Size: {file_size:,} bytes")

            # Ingestion options
            console.print("\n🛠️  [bold yellow]Ingestion Options[/bold yellow]")
            console.print("Choose what to ingest into:")
            console.print("  1. 🗃️  Vector Database only (fast, for search)")
            console.print("  2. 🕸️  Knowledge Graph only (slow, for relationships)")
            console.print("  3. 🚀 Both Vector DB + Knowledge Graph (recommended)")

            ingestion_choice = Prompt.ask(
                "Select ingestion option", choices=["1", "2", "3"], default="3"
            )

            # Set ingestion flags based on choice
            ingest_vector = ingestion_choice in ["1", "3"]
            ingest_graph = ingestion_choice in ["2", "3"]

            # Show selected options with time estimates
            options_text = []
            time_estimate = ""
            if ingest_vector and ingest_graph:
                options_text.append("🗃️ Vector Database")
                options_text.append("🕸️ Knowledge Graph")
                time_estimate = " (may take 1-3 minutes)"
            elif ingest_vector:
                options_text.append("🗃️ Vector Database")
                time_estimate = " (typically 10-30 seconds)"
            elif ingest_graph:
                options_text.append("🕸️ Knowledge Graph")
                time_estimate = " (may take 1-2 minutes)"

            console.print(f"📝 Selected: {' + '.join(options_text)}{time_estimate}")

            # Confirm upload
            if not Confirm.ask(
                f"Upload '{filename}' to tenant {self.current_tenant_id}?"
            ):
                console.print("[yellow]Upload cancelled[/yellow]")
                return

            # Show progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_description = "Uploading and processing document"
                if ingest_graph and not ingest_vector:
                    task_description += " (graph processing may take longer)"
                elif ingest_vector and ingest_graph:
                    task_description += " (full processing may take longer)"

                task = progress.add_task(task_description + "...", total=None)

                try:
                    # Read and upload file
                    with open(file_path, "rb") as file:
                        files = {"file": (filename, file, "text/plain")}
                        data = {
                            "ingest_vector": str(ingest_vector).lower(),
                            "ingest_graph": str(ingest_graph).lower(),
                        }

                        response = await self.upload_client.post(
                            f"{self.api_base_url}/documents", files=files, data=data
                        )

                    progress.update(task, completed=True)

                    if response.status_code == 200:
                        upload_result = response.json()

                        console.print(
                            "✅ [bold green]Document uploaded successfully![/bold green]"
                        )

                        # Display upload results
                        table = Table(title="Upload Results")
                        table.add_column("Property", style="cyan")
                        table.add_column("Value", style="green")

                        table.add_row("Document ID", upload_result["document_id"])
                        table.add_row("Filename", upload_result["filename"])
                        table.add_row("Tenant ID", upload_result["tenant_id"])
                        table.add_row("Status", upload_result["status"])
                        table.add_row("Uploaded At", str(upload_result["uploaded_at"]))
                        table.add_row(
                            "Chunks Created", str(upload_result["chunks_created"])
                        )
                        table.add_row(
                            "Processing Time",
                            f"{upload_result['processing_time_ms']:.1f}ms",
                        )
                        table.add_row(
                            "Vector Stored",
                            "✅" if upload_result["vector_stored"] else "❌",
                        )
                        table.add_row(
                            "Graph Stored",
                            "✅" if upload_result["graph_stored"] else "❌",
                        )

                        console.print(table)

                        # Show what was actually processed
                        processed_text = []
                        if upload_result["vector_stored"]:
                            processed_text.append("🗃️ vector database")
                        if upload_result["graph_stored"]:
                            processed_text.append("🕸️ knowledge graph")

                        if processed_text:
                            console.print(
                                f"\n🎉 [bold green]Document processed into: {' + '.join(processed_text)}[/bold green]"
                            )
                            console.print(
                                "📊 [dim]Your document is now available for search and chat![/dim]"
                            )
                        else:
                            console.print(
                                "\n⚠️  [yellow]Warning: Document uploaded but no processing completed[/yellow]"
                            )

                    else:
                        error_detail = response.json().get(
                            "detail", "Document upload failed"
                        )
                        console.print(f"❌ [bold red]{error_detail}[/bold red]")

                except httpx.TimeoutException:
                    console.print("⏱️ [bold yellow]Upload timed out[/bold yellow]")
                    console.print(
                        "[dim]Knowledge graph processing can take several minutes.[/dim]"
                    )
                    console.print(
                        "[dim]Your document may still be processing in the background.[/dim]"
                    )
                except httpx.RequestError as e:
                    console.print(f"❌ [bold red]Network error: {e}[/bold red]")
                except Exception as e:
                    console.print(f"❌ [bold red]Upload error: {e}[/bold red]")

        except Exception as e:
            console.print(f"❌ [bold red]Error: {e}[/bold red]")

    async def main_menu(self):
        """Main interactive menu."""
        while True:
            try:
                # Show current status
                status = (
                    f"Authenticated as {self.current_user_id} ({self.current_tenant_id})"
                    if self.jwt_token
                    else "Not authenticated"
                )

                console.print(
                    f"\n🏠 [bold blue]Multi-Tenant RAG CLI (HTTP)[/bold blue] - {status}"
                )
                console.print("\nSelect an option:")
                console.print("  1. 🔐 Authenticate")
                console.print("  2. ➕ Create New Tenant")
                console.print("  3. ℹ️  Show Tenant Info")
                console.print("  4. 📄 Upload Document (with ingestion options)")
                console.print("  5. 🔍 Advanced Search (Technical)")
                console.print("  6. 💬 Chat Mode")
                console.print("  7. 🏥 API Health Check")
                console.print("  8. 🚪 Exit")

                choice = Prompt.ask(
                    "Enter choice", choices=["1", "2", "3", "4", "5", "6", "7", "8"]
                )

                if choice == "1":
                    await self.authenticate()
                elif choice == "2":
                    await self.create_tenant()
                elif choice == "3":
                    await self.show_tenant_info()
                elif choice == "4":
                    await self.upload_document()
                elif choice == "5":
                    await self.search_menu()
                elif choice == "6":
                    await self.chat_mode()
                elif choice == "7":
                    await self.check_api_health()
                elif choice == "8":
                    console.print("👋 [yellow]Goodbye![/yellow]")
                    break

            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"❌ [bold red]Error: {e}[/bold red]")


@click.command()
@click.option("--api-url", default="http://localhost:8000", help="API server URL")
@click.option("--tenant-id", help="Pre-authenticate with tenant ID")
@click.option("--api-key", help="API key for authentication")
@click.option("--user-id", default="cli_user", help="User ID for authentication")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(
    api_url: str = "http://localhost:8000",
    tenant_id: Optional[str] = None,
    api_key: Optional[str] = None,
    user_id: str = "cli_user",
    debug: bool = False,
):
    """HTTP-based Interactive Multi-Tenant CLI for Hybrid RAG System."""

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    async def run_cli():
        async with HTTPMultiTenantCLI(api_base_url=api_url) as cli:
            try:
                # Check API health first
                console.print("🔍 [blue]Checking API server health...[/blue]")
                if not await cli.check_api_health():
                    console.print(
                        "❌ [bold red]Cannot connect to API server. Please ensure the API is running.[/bold red]"
                    )
                    console.print(f"   Expected URL: {api_url}")
                    console.print(
                        "   Start the API with: python interactive_multi_tenant_api.py"
                    )
                    return

                # Pre-authenticate if credentials provided
                if tenant_id and api_key:
                    try:
                        # Set credentials for automatic authentication
                        auth_success = await cli.authenticate_with_credentials(
                            tenant_id, api_key, user_id
                        )
                        if auth_success:
                            console.print(
                                f"✅ [green]Pre-authenticated for tenant {tenant_id}[/green]"
                            )
                        else:
                            console.print("❌ [red]Pre-authentication failed[/red]")
                    except Exception as e:
                        console.print(f"❌ [red]Pre-authentication error: {e}[/red]")

                await cli.main_menu()

            except Exception as e:
                console.print(f"❌ [bold red]CLI error: {e}[/bold red]")
                logger.exception("CLI error")

    # Add method for programmatic authentication
    async def authenticate_with_credentials(
        self, tenant_id: str, api_key: str, user_id: str
    ) -> bool:
        """Authenticate programmatically with provided credentials."""
        try:
            auth_data = {"tenant_id": tenant_id, "api_key": api_key, "user_id": user_id}

            response = await self.client.post(
                f"{self.api_base_url}/auth/login", json=auth_data
            )

            if response.status_code == 200:
                auth_result = response.json()
                self.jwt_token = auth_result["access_token"]
                self.current_tenant_id = tenant_id
                self.current_user_id = user_id

                # Update client headers with token
                auth_header = {"Authorization": f"Bearer {self.jwt_token}"}
                self.client.headers.update(auth_header)
                self.upload_client.headers.update(auth_header)
                return True
            else:
                return False

        except Exception:
            return False

    # Monkey-patch the method
    HTTPMultiTenantCLI.authenticate_with_credentials = authenticate_with_credentials

    # Welcome message
    console.print(
        Panel.fit(
            "🚀 [bold blue]Multi-Tenant RAG CLI (HTTP API)[/bold blue]\n"
            "Interactive command-line interface using HTTP API endpoints\n"
            "Supports JWT authentication, vector search, graph search, hybrid search, and chat mode\n"
            f"API Server: {api_url}",
            border_style="blue",
        )
    )

    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
