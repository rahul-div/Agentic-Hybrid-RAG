"""
Robust Gemini Client - Drop-in replacement for Graphiti's GeminiClient

This client guarantees valid JSON responses by:
- Forcing temperature=0.0 (deterministic output)
- Forcing response_mime_type="application/json"
- Adding post-processing to salvage minor formatting issues from Gemini
- Providing structured retry logic with explicit JSON constraints
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

    GRAPHITI_AVAILABLE = True
except ImportError:
    # Fallback for development
    GRAPHITI_AVAILABLE = False

    class GeminiClient:
        def __init__(self, *args, **kwargs):
            pass

    class LLMConfig:
        def __init__(self, *args, **kwargs):
            pass


logger = logging.getLogger(__name__)


class RobustGeminiClient(GeminiClient):
    """
    Drop-in replacement for Graphiti's GeminiClient that:
    - Forces temperature=0.0 (deterministic output)
    - Forces response_mime_type="application/json"
    - Adds post-processing to salvage minor formatting issues from Gemini
      before returning to Graphiti's downstream structured parser
    """

    def __init__(self, config: LLMConfig):
        """Initialize with forced deterministic settings."""
        # Override temperature to 0.0 for deterministic output
        if hasattr(config, "temperature"):
            config.temperature = 0.0
        super().__init__(config)

        # Set our own max_retries since LLMConfig doesn't support it
        self.max_retries = 2  # Conservative retry for production reliability
        logger.info(
            "RobustGeminiClient initialized with temperature=0.0 for deterministic JSON output"
        )

    def _clean_raw_json_text(self, raw_text: str) -> str:
        """
        Try to coerce malformed Gemini output into valid JSON:
        - Strip code fences ```json ... ```
        - Collapse excessive newlines inside quoted strings
        - Trim trailing commas
        - Handle unterminated strings
        """
        txt = raw_text.strip()

        # Remove ```json ... ``` or ``` ... ``` wrappers if present
        if "```" in txt:
            # Take inside the first fenced block if found
            parts = txt.split("```")
            # Heuristic: prefer the longest section that looks like JSON-ish
            candidates = [p for p in parts if "{" in p and "}" in p]
            if candidates:
                txt = max(candidates, key=len)

        # Sometimes Gemini hallucinates 'Here is the JSON:' before the object
        first_brace = txt.find("{")
        last_brace = txt.rfind("}")
        if first_brace != -1 and last_brace != -1:
            txt = txt[first_brace : last_brace + 1]

        # Handle the specific "unterminated string" issue from your logs
        # Remove excessive newlines that break JSON strings
        txt = txt.replace("\r", "")

        # Fix unterminated strings by looking for patterns like:
        # "Google\n\n\n\n\n... (thousands of newlines)
        # and replacing with just "Google"
        import re

        txt = re.sub(r'"\s*\n+\s*$', '"', txt)  # Fix unterminated strings at end
        txt = re.sub(r'"\s*\n+\s*(?=[,}\]])', '"', txt)  # Fix before punctuation

        # Collapse excessive newlines in general
        txt = re.sub(r"\n{3,}", "\n", txt)

        # Remove trailing commas before closing braces/brackets
        txt = re.sub(r",\s*([}\]])", r"\1", txt)

        return txt.strip()

    def _try_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempt strict parse, then salvage.
        Raises if completely hopeless.
        """
        # First try strict parse
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("RobustGeminiClient: Initial JSON parse failed: %s", e)

        # Try salvage cleaning
        cleaned = self._clean_raw_json_text(raw_text)
        try:
            result = json.loads(cleaned)
            logger.info("RobustGeminiClient: Successfully salvaged JSON after cleaning")
            return result
        except json.JSONDecodeError as e:
            logger.error(
                "RobustGeminiClient: Failed to parse JSON even after salvage cleaning"
            )
            logger.error("Original text length: %d", len(raw_text))
            logger.error("Cleaned text length: %d", len(cleaned))
            logger.error(
                "Cleaned text preview: %s",
                cleaned[:500] + "..." if len(cleaned) > 500 else cleaned,
            )
            raise e

    async def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Replacement for GeminiClient.generate_structured.
        We:
        - Force deterministic generation config
        - Force response_mime_type=application/json
        - Retry with schema reminder if parsing fails
        """
        last_err = None
        base_reminder = (
            "Return ONLY valid JSON. "
            "Do not include comments, markdown, code fences, or trailing commas. "
            "Do not include newlines inside string values unless they are \\n. "
            "Do not include explanations or prose. "
            "Close all quotes and braces properly."
        )

        for attempt in range(self.max_retries):
            # For retry attempts, append extra "previous response was invalid" message
            attempt_messages = list(messages)
            if attempt > 0:
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid JSON. "
                            "Please try again. " + base_reminder
                        ),
                    }
                )
            else:
                # Add JSON constraint to first attempt too
                if attempt_messages and attempt_messages[-1]["role"] == "user":
                    attempt_messages[-1]["content"] += "\n\n" + base_reminder

            try:
                # Call parent's raw generation method with forced JSON config
                raw_text = await self._raw_generate_json(
                    attempt_messages, response_schema
                )
                return self._try_parse_json(raw_text)
            except Exception as e:
                last_err = e
                logger.warning(
                    "RobustGeminiClient: parsing failed on attempt %s/%s: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

        logger.error(
            "RobustGeminiClient: all retries exhausted. Last error: %s", last_err
        )
        raise last_err

    async def _raw_generate_json(
        self,
        messages: List[Dict[str, Any]],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Low-level Gemini call with forced JSON configuration.
        We force:
        - temperature=0.0
        - response_mime_type='application/json'
        - max_output_tokens large enough
        """
        try:
            # Convert messages to Gemini format if needed
            gemini_contents = []
            for msg in messages:
                if msg["role"] == "user":
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": msg["content"]}]}
                    )
                elif msg["role"] == "system":
                    # Prepend system message to first user message
                    if gemini_contents and gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"][0]["text"] = (
                            msg["content"]
                            + "\n\n"
                            + gemini_contents[-1]["parts"][0]["text"]
                        )
                    else:
                        gemini_contents.append(
                            {"role": "user", "parts": [{"text": msg["content"]}]}
                        )

            # Use the parent client's generate method with forced JSON config
            if hasattr(self, "client") and hasattr(self.client, "generate_content"):
                response = await self.client.generate_content(
                    contents=gemini_contents,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",
                    },
                )
                return response.text
            else:
                # Fallback: call the parent's generate method
                # This may vary based on Graphiti version
                return await super().generate(
                    messages,
                    **{
                        "temperature": 0.0,
                        "max_tokens": 4096,
                        "response_format": {"type": "json_object"},
                    },
                )

        except Exception as e:
            logger.error("RobustGeminiClient: Raw generation failed: %s", e)
            raise

    async def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Override generate to ensure JSON mode for all calls."""
        # Force JSON-friendly parameters
        kwargs.update(
            {
                "temperature": 0.0,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }
        )

        try:
            result = await super().generate(messages, **kwargs)
            # Validate that result is JSON-parseable if it should be
            if any("json" in msg.get("content", "").lower() for msg in messages):
                try:
                    json.loads(result)
                except json.JSONDecodeError:
                    # Apply cleaning if this should be JSON
                    result = self._clean_raw_json_text(result)
                    json.loads(result)  # Validate again
            return result
        except Exception as e:
            logger.error("RobustGeminiClient: Generate method failed: %s", e)
            raise
