"""
ResponseFieldExtractor - Centralized response field extraction for LiteLLM providers.

This utility solves the problem of if/elif chains that block simultaneous extraction
of thinking, content, and tool_calls fields. Each field is extracted independently.

Usage:
    from litellm.litellm_core_utils.response_field_extractor import ResponseFieldExtractor

    extracted = ResponseFieldExtractor.extract_all(
        message=response_json_message,
        provider="ollama"
    )

    # All fields extracted independently
    extracted.content          # str or None
    extracted.reasoning_content # str or None
    extracted.tool_calls       # List[Dict] in OpenAI format or None
    extracted.finish_reason    # "tool_calls" if tool_calls present, else "stop"

Related issues:
- https://github.com/BerriAI/litellm/issues/18922 (qwen3 tool_calls dropped)
- https://github.com/BerriAI/litellm/issues/18926 (opus thinking dropped)
- https://github.com/BerriAI/litellm/issues/18787 (Bedrock Claude thinking+tools)
- https://github.com/BerriAI/litellm/issues/18484 (Gemini <think> tags)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from litellm._uuid import uuid


@dataclass
class ExtractedResponseFields:
    """
    Container for all extracted response fields.
    All fields are extracted independently and can coexist.
    """

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Always in OpenAI format
    finish_reason: str = "stop"

    # Provider-specific fields that don't need transformation
    provider_specific_fields: Dict[str, Any] = field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        """Check if tool_calls are present and non-empty."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def has_thinking(self) -> bool:
        """Check if reasoning/thinking content is present."""
        return (
            self.reasoning_content is not None
            or (self.thinking_blocks is not None and len(self.thinking_blocks) > 0)
        )


class ResponseFieldExtractor:
    """
    Utility for extracting and normalizing response fields from various LLM providers.

    Design principles:
    1. Extract ALL fields independently (no if/elif blocking)
    2. Normalize to OpenAI format
    3. Handle provider-specific quirks transparently
    """

    # Compiled regex patterns for performance
    # Matches ALL <think>...</think> blocks (for extraction)
    _THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # Matches <think>...</think> blocks with optional trailing whitespace (for removal)
    _THINK_TAG_REMOVAL_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    # Known fields that are handled specially
    KNOWN_FIELDS = {
        "content",
        "role",
        "reasoning_content",
        "reasoning",
        "thinking",
        "thinking_blocks",
        "tool_calls",
        "function_call",
        "finish_reason",
        "stop_reason",
        "done_reason",
    }

    @staticmethod
    def extract_all(
        message: Optional[Dict[str, Any]],
        provider: str = "openai",
    ) -> ExtractedResponseFields:
        """
        Main entry point. Extracts all fields from a provider's message response.

        Args:
            message: The message dict from the provider response
            provider: One of 'openai', 'ollama', 'anthropic', 'snowflake', etc.

        Returns:
            ExtractedResponseFields with all fields populated independently
        """
        if message is None:
            return ExtractedResponseFields()

        result = ExtractedResponseFields()

        # 1. Extract thinking/reasoning FIRST (before content, as content may contain <think> tags)
        result.reasoning_content, result.thinking_blocks = (
            ResponseFieldExtractor._extract_thinking(message, provider)
        )

        # 2. Extract content (may need to strip <think> tags if thinking was extracted from it)
        result.content = ResponseFieldExtractor._extract_content(
            message, provider, result.reasoning_content
        )

        # 3. Extract tool_calls (always, independently, normalized to OpenAI format)
        result.tool_calls = ResponseFieldExtractor._extract_tool_calls(message, provider)

        # 4. Determine finish_reason based on extracted fields
        result.finish_reason = ResponseFieldExtractor._determine_finish_reason(
            message, result
        )

        # 5. Collect provider-specific fields that weren't handled
        result.provider_specific_fields = ResponseFieldExtractor._extract_provider_specific(
            message
        )

        return result

    @staticmethod
    def _extract_content(
        message: Dict[str, Any],
        provider: str,
        extracted_reasoning: Optional[str],
    ) -> Optional[str]:
        """
        Extract text content, handling provider-specific formats.

        If reasoning was extracted from <think> tags in content, this removes those tags.
        """
        content = message.get("content")

        if content is None:
            return None

        if isinstance(content, str):
            # If we extracted reasoning from <think> tags, strip them from content
            if extracted_reasoning and "<think>" in content:
                # Remove ALL <think>...</think> blocks from content
                content = ResponseFieldExtractor._THINK_TAG_REMOVAL_PATTERN.sub("", content)
                content = content.strip()
                return content if content else None
            return content

        # Handle Anthropic-style content lists
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts) if text_parts else None

        return None

    @staticmethod
    def _extract_thinking(
        message: Dict[str, Any],
        provider: str,
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Extract reasoning/thinking content from message.

        Returns:
            Tuple of (reasoning_content_string, thinking_blocks_list)
        """
        reasoning_content: Optional[str] = None
        thinking_blocks: Optional[List[Dict[str, Any]]] = None

        # Check for explicit reasoning_content field (already normalized)
        if "reasoning_content" in message and message["reasoning_content"]:
            reasoning_content = message["reasoning_content"]

        # Check for 'reasoning' field (some providers use this)
        if "reasoning" in message and message["reasoning"] and reasoning_content is None:
            reasoning_content = message["reasoning"]

        # Check for 'thinking' field (Ollama/qwen3 style)
        if "thinking" in message and message["thinking"]:
            thinking_value = message["thinking"]
            if isinstance(thinking_value, str):
                if reasoning_content is None:
                    reasoning_content = thinking_value
            elif isinstance(thinking_value, dict):
                # Convert to thinking_blocks format
                thinking_blocks = [thinking_value]
                if reasoning_content is None:
                    reasoning_content = thinking_value.get("thinking")

        # Check for thinking_blocks (Anthropic style)
        if "thinking_blocks" in message and message["thinking_blocks"]:
            blocks = message["thinking_blocks"]
            if isinstance(blocks, list) and len(blocks) > 0:
                thinking_blocks = blocks
                if reasoning_content is None:
                    # Concatenate thinking text from blocks
                    texts = []
                    for block in blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking" and block.get("thinking"):
                                texts.append(block["thinking"])
                    if texts:
                        reasoning_content = "\n".join(texts)

        # Parse <think> tags from content if no explicit thinking found
        content = message.get("content")
        if reasoning_content is None and isinstance(content, str):
            if "<think>" in content and "</think>" in content:
                # Extract ALL <think> blocks and concatenate them
                matches = ResponseFieldExtractor._THINK_TAG_PATTERN.findall(content)
                if matches:
                    reasoning_content = "\n".join(m.strip() for m in matches)

        return reasoning_content, thinking_blocks

    @staticmethod
    def _extract_tool_calls(
        message: Dict[str, Any],
        provider: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract and normalize tool_calls to OpenAI format.

        OpenAI format:
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "function_name",
                "arguments": '{"key": "value"}'  # JSON string, not dict
            }
        }
        """
        raw_tool_calls = message.get("tool_calls")

        if raw_tool_calls is None:
            return None

        if not isinstance(raw_tool_calls, list):
            return None

        if len(raw_tool_calls) == 0:
            return None

        normalized = []
        for idx, tc in enumerate(raw_tool_calls):
            normalized_tc = ResponseFieldExtractor._normalize_single_tool_call(
                tc, provider, idx
            )
            if normalized_tc:
                normalized.append(normalized_tc)

        return normalized if normalized else None

    @staticmethod
    def _normalize_single_tool_call(
        tool_call: Dict[str, Any],
        provider: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize a single tool call to OpenAI format.

        Handles formats from:
        - OpenAI (already correct format)
        - Ollama (no id, arguments as dict)
        - Anthropic (tool_use with input)
        - Snowflake (nested tool_use)
        """
        if not isinstance(tool_call, dict):
            return None

        # Case 1: Already in OpenAI format (has function and id)
        if "function" in tool_call and "id" in tool_call:
            func = tool_call["function"]
            arguments = func.get("arguments", "{}")

            # Ensure arguments is a JSON string
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            return {
                "id": tool_call["id"],
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": func.get("name", ""),
                    "arguments": arguments,
                },
            }

        # Case 2: Ollama format (has function but no id, arguments as dict)
        if "function" in tool_call and "id" not in tool_call:
            func = tool_call["function"]
            arguments = func.get("arguments", {})

            # Convert dict arguments to JSON string
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            return {
                "id": tool_call.get("id", f"call_{uuid.uuid4()}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": arguments,
                },
            }

        # Case 3: Anthropic style (type: tool_use with input)
        if tool_call.get("type") == "tool_use":
            input_data = tool_call.get("input", {})
            if not isinstance(input_data, str):
                input_data = json.dumps(input_data)

            return {
                "id": tool_call.get("id", f"call_{uuid.uuid4()}"),
                "type": "function",
                "function": {
                    "name": tool_call.get("name", ""),
                    "arguments": input_data,
                },
            }

        # Case 4: Snowflake nested tool_use
        if "tool_use" in tool_call:
            tool_use = tool_call["tool_use"]
            input_data = tool_use.get("input", {})
            if not isinstance(input_data, str):
                input_data = json.dumps(input_data)

            return {
                "id": tool_use.get("tool_use_id", f"call_{uuid.uuid4()}"),
                "type": "function",
                "function": {
                    "name": tool_use.get("name", ""),
                    "arguments": input_data,
                },
            }

        return None

    @staticmethod
    def _determine_finish_reason(
        message: Dict[str, Any],
        extracted: ExtractedResponseFields,
    ) -> str:
        """
        Determine the correct finish_reason.

        Priority:
        1. tool_calls present → "tool_calls"
        2. Provider-specified finish_reason → mapped value
        3. Default → "stop"
        """
        # If tool_calls present, finish_reason should be "tool_calls"
        if extracted.has_tool_calls():
            return "tool_calls"

        # Check for provider-specified finish_reason
        for key in ["finish_reason", "stop_reason", "done_reason"]:
            if key in message and message[key]:
                reason = message[key]
                # Map common finish reasons to OpenAI values
                reason_map = {
                    "end_turn": "stop",
                    "stop_sequence": "stop",
                    "max_tokens": "length",
                    "length": "length",
                    "tool_use": "tool_calls",
                    "function_call": "tool_calls",
                }
                return reason_map.get(reason, reason)

        return "stop"

    @staticmethod
    def _extract_provider_specific(
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract provider-specific fields that don't need normalization.

        These are passed through for providers that need access to extra fields.
        """
        return {
            k: v
            for k, v in message.items()
            if k not in ResponseFieldExtractor.KNOWN_FIELDS and v is not None
        }
