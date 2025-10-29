"""
Graphiti Ingestion Manager - Reliability layer for knowledge graph ingestion

This manager provides:
- Per-episode atomic ingestion
- Text sanitization to prevent JSON parsing errors
- Structured error handling and retry capability
- Tenant namespace isolation
- Comprehensive logging and monitoring
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

try:
    from graphiti_core.graphiti import Graphiti
    from graphiti_core.nodes import EpisodeType

    GRAPHITI_AVAILABLE = True
except ImportError:
    # Fallback for development
    GRAPHITI_AVAILABLE = False

    class Graphiti:
        def __init__(self, *args, **kwargs):
            pass

        async def add_episode(self, **kwargs):
            pass

    class EpisodeType:
        text = "text"


logger = logging.getLogger(__name__)


def sanitize_chunk_for_kg(text: str, max_len: int = 1500) -> str:
    """
    Pre-clean text before we hand it to Graphiti/Gemini.

    Why this is critical:
      - Emojis, markdown headings, bold markers cause Gemini to generate malformed JSON
      - Multi-line bullet points and fancy unicode punctuation break JSON strings
      - Long content overwhelms Gemini's context window

    We keep semantics but remove decoration/noise that causes JSON parsing failures.

    Note: max_len should match the performance config max_content_length (1500).
    Since chunks are typically 800 chars, this mainly handles edge cases.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove markdown bold/italics/headings that cause JSON string issues
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold** -> bold
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__ -> bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic* -> italic
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_ -> italic
    text = re.sub(r"#{1,6}\s*", "", text)  # Remove markdown headers
    text = re.sub(r"`([^`]+)`", r"\1", text)  # Remove code backticks

    # Remove emojis and pictographs that cause encoding issues in JSON
    # This includes the 📅, 🧠, 📚 etc. from your temporal_rag_test_story.md
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        " ",
        text,
    )

    # Normalize unicode quotes/dashes that sometimes become garbage in JSON
    text = text.replace("'", "'").replace("'", "'")  # Smart quotes to simple
    text = text.replace(""", '"').replace(""", '"')  # Smart double quotes
    text = text.replace("–", "-").replace("—", "-")  # Em/en dashes to hyphen
    text = text.replace("…", "...")  # Ellipsis to three dots

    # Remove bullet points and list markers that can confuse entity extraction
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse excessive whitespace that can break JSON formatting
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
    text = re.sub(r"\n\s*\n+", "\n", text)  # Multiple newlines to single
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # Trim line edges

    # Remove or replace characters that commonly break JSON parsing
    text = text.replace("\r", "")  # Remove carriage returns
    text = text.replace("\t", " ")  # Tabs to spaces
    text = re.sub(
        r"[^\x20-\x7E\n]", "", text
    )  # Remove non-printable chars except newline

    # Truncate if absurdly long (Gemini context safety)
    if len(text) > max_len:
        # Try to find a good breaking point (sentence boundary)
        truncated = text[:max_len]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        break_point = max(last_period, last_newline)
        if break_point > max_len * 0.7:  # Only use break point if it's not too early
            text = text[: break_point + 1]
        else:
            text = text[:max_len] + "..."

    result = text.strip()

    # Final validation - ensure we have clean, non-empty text
    if not result or len(result) < 10:
        return "Document content"  # Fallback for empty/tiny chunks

    return result


class GraphitiIngestionManager:
    """
    Reliability layer for knowledge graph ingestion.

    Responsibilities:
    - Take pre-chunked docs (already embedded & stored in pgvector)
    - Split them into safe "episodes"
    - For each episode:
        1. sanitize text to prevent JSON parsing errors
        2. call graphiti.add_episode() with proper namespace/group_id
        3. wrap each call in try/except so one bad episode doesn't poison the whole batch
    - Return (success_count, failed_episodes[]) for retry queue
    """

    def __init__(self, graphiti: Graphiti, max_content_length: int = 1500):
        self.graphiti = graphiti
        self.max_content_length = max_content_length
        logger.info(
            f"GraphitiIngestionManager initialized for reliable episode ingestion (max_content_length={max_content_length})"
        )

    async def ingest_chunks_as_episodes(
        self,
        *,
        tenant_id: str,
        doc_name: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Ingest document chunks as individual episodes with full error isolation.

        Args:
            tenant_id: Unique tenant identifier for namespace isolation
            doc_name: Document name for episode naming and tracking
            chunks: List of chunk dictionaries with structure:
                [{
                    "chunk_id": "uuid_string",
                    "text": "raw chunk text content",
                    "created_at": datetime (optional),
                    "metadata": dict (optional)
                }, ...]

        Returns:
            Tuple of (num_successful_episodes, failed_episode_records)

        Each episode is processed atomically - success or failure is isolated
        so partial ingestion doesn't corrupt the knowledge graph.
        """
        if not chunks:
            logger.warning("No chunks provided for ingestion")
            return 0, []

        namespace = f"tenant_{tenant_id}"
        successes = 0
        failures: List[Dict[str, Any]] = []

        logger.info(
            "Starting KG ingestion for tenant %s: %d chunks from document '%s'",
            tenant_id,
            len(chunks),
            doc_name,
        )  # Process each chunk as a separate episode for atomic success/failure
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            raw_text = chunk.get("text", "")

            if not raw_text:
                logger.warning(
                    "Skipping empty chunk %s for tenant %s", chunk_id, tenant_id
                )
                continue

            # Sanitize text to prevent Gemini JSON parsing errors
            safe_text = sanitize_chunk_for_kg(raw_text, self.max_content_length)

            # Build episode body with clear structure
            episode_body = self._build_episode_body(doc_name, idx + 1, safe_text)

            # Generate temporal anchoring
            reference_time = chunk.get("created_at") or datetime.now(timezone.utc)

            # Generate unique episode name for tracking
            episode_name = self._generate_episode_name(
                namespace, chunk_id, reference_time
            )

            try:
                # Attempt atomic episode ingestion
                await self._ingest_single_episode(
                    episode_name=episode_name,
                    group_id=namespace,
                    reference_time=reference_time,
                    episode_body=episode_body,
                )
                successes += 1
                logger.debug("✅ Episode ingested: %s", episode_name)

            except Exception as e:
                # Isolate failure - don't let it break other episodes
                logger.error(
                    "❌ Episode ingest failed for tenant %s chunk_id=%s: %s",
                    tenant_id,
                    chunk_id,
                    str(e),
                )

                # Record structured failure info for retry queue
                failure_record = {
                    "tenant_id": tenant_id,
                    "chunk_id": chunk_id,
                    "episode_name": episode_name,
                    "doc_name": doc_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "episode_body": episode_body,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "retryable": self._is_retryable_error(e),
                }
                failures.append(failure_record)

        # Log final summary
        logger.info(
            "KG ingestion completed for tenant %s: %d/%d episodes succeeded, %d failed",
            tenant_id,
            successes,
            len(chunks),
            len(failures),
        )

        if failures:
            retryable_count = sum(1 for f in failures if f["retryable"])
            logger.warning(
                "Tenant %s has %d failed episodes (%d retryable) that need attention",
                tenant_id,
                len(failures),
                retryable_count,
            )

        return successes, failures

    def _build_episode_body(
        self, doc_name: str, section_num: int, safe_text: str
    ) -> str:
        """
        Build well-structured episode body for consistent entity extraction.

        Format ensures Gemini gets clear context for entity/relationship extraction
        while keeping within token limits.
        """
        return f"Document: {doc_name}\nSection {section_num}:\n{safe_text}"

    def _generate_episode_name(
        self, namespace: str, chunk_id: str, reference_time: datetime
    ) -> str:
        """Generate unique, trackable episode name."""
        timestamp = int(reference_time.timestamp())
        # Truncate chunk_id if too long to keep episode names manageable
        short_chunk_id = chunk_id[:8] if len(chunk_id) > 8 else chunk_id
        return f"{namespace}_{short_chunk_id}_{timestamp}"

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is worth retrying.

        Network errors, timeouts, and temporary API issues are retryable.
        JSON parsing errors and authentication issues typically are not.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Retryable conditions
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "unavailable",
            "rate limit",
            "quota",
            "temporary",
            "503",
            "502",
            "504",
        ]

        # Non-retryable conditions
        non_retryable_indicators = [
            "authentication",
            "authorization",
            "permission",
            "401",
            "403",
            "json",
            "parse",
            "invalid",
            "malformed",
        ]

        # Check for non-retryable first (more specific)
        if any(indicator in error_str for indicator in non_retryable_indicators):
            return False

        # Check for retryable
        if any(indicator in error_str for indicator in retryable_indicators):
            return True

        # Default: retry on unknown errors (conservative approach)
        return True

    async def _ingest_single_episode(
        self,
        *,
        episode_name: str,
        group_id: str,
        reference_time: datetime,
        episode_body: str,
    ) -> None:
        """
        Execute atomic episode ingestion with Graphiti.

        This is the critical method that calls Graphiti.add_episode()
        with consistent parameters and proper error propagation.
        """
        logger.debug(
            "📚 Ingesting KG episode %s (namespace=%s, body_length=%d)",
            episode_name,
            group_id,
            len(episode_body),
        )

        # Ensure reference_time is timezone-aware UTC
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        try:
            # Call Graphiti with sanitized, well-structured data
            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=EpisodeType.text,
                source_description="multi_tenant_rag_ingestion",
                reference_time=reference_time,
                group_id=group_id,  # Tenant namespace isolation
            )

            logger.info("✅ Episode ingested successfully: %s", episode_name)

        except Exception as e:
            # Re-raise with additional context for debugging
            logger.error(
                "Failed to ingest episode %s: %s (body preview: %s...)",
                episode_name,
                str(e),
                episode_body[:100],
            )
            raise

    async def get_tenant_episode_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get statistics about episodes for a tenant.
        Useful for monitoring and debugging ingestion health.
        """
        try:
            namespace = f"tenant_{tenant_id}"

            # Search for episodes in this tenant's namespace
            # This is a basic implementation - you might want to use
            # more specific Graphiti queries if available
            results = await self.graphiti.search(
                query="*",
                group_ids=[namespace],
            )

            stats = {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "total_episodes": len(results),
                "last_checked": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "Retrieved episode stats for tenant %s: %d episodes",
                tenant_id,
                len(results),
            )
            return stats

        except Exception as e:
            logger.error("Failed to get episode stats for tenant %s: %s", tenant_id, e)
            return {
                "tenant_id": tenant_id,
                "error": str(e),
                "total_episodes": 0,
            }

    async def retry_failed_episodes(
        self, failed_episodes: List[Dict[str, Any]]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Retry a batch of previously failed episodes.

        Args:
            failed_episodes: List of failure records from previous ingestion

        Returns:
            Tuple of (newly_successful_count, still_failed_episodes)
        """
        if not failed_episodes:
            return 0, []

        successes = 0
        still_failed = []

        logger.info("Retrying %d failed episodes", len(failed_episodes))

        for failure_record in failed_episodes:
            if not failure_record.get("retryable", True):
                # Skip non-retryable errors
                still_failed.append(failure_record)
                continue

            try:
                await self._ingest_single_episode(
                    episode_name=failure_record["episode_name"],
                    group_id=f"tenant_{failure_record['tenant_id']}",
                    reference_time=datetime.now(
                        timezone.utc
                    ),  # Use current time for retry
                    episode_body=failure_record["episode_body"],
                )
                successes += 1
                logger.info("✅ Retry successful: %s", failure_record["episode_name"])

            except Exception as e:
                # Update failure record with retry info
                failure_record["last_retry_at"] = datetime.now(timezone.utc).isoformat()
                failure_record["last_retry_error"] = str(e)
                failure_record["retry_count"] = failure_record.get("retry_count", 0) + 1
                still_failed.append(failure_record)

                logger.error(
                    "❌ Retry failed for %s: %s", failure_record["episode_name"], str(e)
                )

        logger.info(
            "Retry completed: %d newly successful, %d still failed",
            successes,
            len(still_failed),
        )

        return successes, still_failed
