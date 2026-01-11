"""
Tests for ResponseFieldExtractor utility.

These tests verify that all response fields (content, thinking, tool_calls)
are extracted INDEPENDENTLY without if/elif blocking.

Related issues:
- https://github.com/BerriAI/litellm/issues/18922 (qwen3 tool_calls dropped)
- https://github.com/BerriAI/litellm/issues/18926 (opus thinking dropped)
"""

import json
import pytest

from litellm.litellm_core_utils.response_field_extractor import (
    ExtractedResponseFields,
    ResponseFieldExtractor,
)


class TestExtractedResponseFields:
    """Tests for the ExtractedResponseFields dataclass."""

    def test_has_tool_calls_with_tools(self):
        """Test has_tool_calls returns True when tool_calls present."""
        fields = ExtractedResponseFields(
            tool_calls=[{"id": "call_123", "type": "function", "function": {}}]
        )
        assert fields.has_tool_calls() is True

    def test_has_tool_calls_empty_list(self):
        """Test has_tool_calls returns False for empty list."""
        fields = ExtractedResponseFields(tool_calls=[])
        assert fields.has_tool_calls() is False

    def test_has_tool_calls_none(self):
        """Test has_tool_calls returns False for None."""
        fields = ExtractedResponseFields(tool_calls=None)
        assert fields.has_tool_calls() is False

    def test_has_thinking_with_reasoning_content(self):
        """Test has_thinking returns True when reasoning_content present."""
        fields = ExtractedResponseFields(reasoning_content="Let me think...")
        assert fields.has_thinking() is True

    def test_has_thinking_with_blocks(self):
        """Test has_thinking returns True when thinking_blocks present."""
        fields = ExtractedResponseFields(
            thinking_blocks=[{"type": "thinking", "thinking": "..."}]
        )
        assert fields.has_thinking() is True


class TestResponseFieldExtractorBasic:
    """Basic extraction tests."""

    def test_extract_all_empty_message(self):
        """Test extraction from empty message."""
        result = ResponseFieldExtractor.extract_all({})
        assert result.content is None
        assert result.reasoning_content is None
        assert result.tool_calls is None
        assert result.finish_reason == "stop"

    def test_extract_all_none_message(self):
        """Test extraction from None message."""
        result = ResponseFieldExtractor.extract_all(None)
        assert result.content is None
        assert result.reasoning_content is None
        assert result.tool_calls is None

    def test_extract_content_only(self):
        """Test extraction of content-only message."""
        message = {"role": "assistant", "content": "Hello, world!"}
        result = ResponseFieldExtractor.extract_all(message)
        assert result.content == "Hello, world!"
        assert result.reasoning_content is None
        assert result.tool_calls is None
        assert result.finish_reason == "stop"


class TestThinkingExtraction:
    """Tests for thinking/reasoning extraction."""

    def test_extract_thinking_field(self):
        """Test extraction of 'thinking' field (Ollama/qwen3 style)."""
        message = {
            "role": "assistant",
            "content": "",
            "thinking": "Let me analyze this request...",
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.reasoning_content == "Let me analyze this request..."

    def test_extract_reasoning_content_field(self):
        """Test extraction of 'reasoning_content' field (pre-normalized)."""
        message = {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "I should greet the user.",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.reasoning_content == "I should greet the user."
        assert result.content == "Hello"

    def test_extract_thinking_blocks(self):
        """Test extraction of thinking_blocks (Anthropic style)."""
        message = {
            "role": "assistant",
            "content": "Result",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "Step 1: analyze"},
                {"type": "thinking", "thinking": "Step 2: solve"},
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="anthropic")
        assert result.thinking_blocks is not None
        assert len(result.thinking_blocks) == 2
        assert "Step 1" in result.reasoning_content
        assert "Step 2" in result.reasoning_content

    def test_extract_think_tags_from_content(self):
        """Test extraction of <think> tags from content."""
        message = {
            "role": "assistant",
            "content": "<think>Let me reason about this...</think>Here is the answer.",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.reasoning_content == "Let me reason about this..."
        assert result.content == "Here is the answer."
        assert "<think>" not in result.content


class TestToolCallsExtraction:
    """Tests for tool_calls extraction and normalization."""

    def test_extract_openai_format_tool_calls(self):
        """Test that OpenAI format tool_calls pass through correctly."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc123"
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.tool_calls[0]["function"]["arguments"] == '{"location": "Tokyo"}'
        assert result.finish_reason == "tool_calls"

    def test_extract_ollama_format_tool_calls(self):
        """Test normalization of Ollama format (dict arguments, no id)."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo", "units": "celsius"},
                    }
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

        tc = result.tool_calls[0]
        assert tc["id"].startswith("call_")
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"

        # Arguments should be JSON string, not dict
        assert isinstance(tc["function"]["arguments"], str)
        args = json.loads(tc["function"]["arguments"])
        assert args["location"] == "Tokyo"
        assert args["units"] == "celsius"

    def test_extract_anthropic_tool_use_format(self):
        """Test normalization of Anthropic tool_use format."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "search_db",
                    "input": {"query": "test"},
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="anthropic")
        assert result.tool_calls is not None
        tc = result.tool_calls[0]
        assert tc["id"] == "toolu_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search_db"
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}

    def test_extract_snowflake_nested_tool_use(self):
        """Test normalization of Snowflake nested tool_use format."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "tool_use": {
                        "tool_use_id": "sf_123",
                        "name": "analyze_data",
                        "input": {"dataset": "sales"},
                    }
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="snowflake")
        assert result.tool_calls is not None
        tc = result.tool_calls[0]
        assert tc["id"] == "sf_123"
        assert tc["function"]["name"] == "analyze_data"

    def test_extract_multiple_tool_calls(self):
        """Test extraction of multiple tool_calls."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}},
                {"function": {"name": "get_weather", "arguments": {"location": "NYC"}}},
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.tool_calls[1]["function"]["name"] == "get_weather"
        # Each should have a unique ID
        assert result.tool_calls[0]["id"] != result.tool_calls[1]["id"]

    def test_empty_tool_calls_list(self):
        """Test that empty tool_calls list returns None."""
        message = {"role": "assistant", "content": "Hello", "tool_calls": []}
        result = ResponseFieldExtractor.extract_all(message)
        assert result.tool_calls is None
        assert result.finish_reason == "stop"


class TestSimultaneousExtraction:
    """
    Critical tests: verify that thinking + tool_calls can be extracted simultaneously.
    This is the core bug fix - the if/elif pattern was blocking this.
    """

    def test_qwen3_thinking_and_tool_calls_simultaneously(self):
        """
        Test the exact qwen3 scenario that was broken.
        Issue: https://github.com/BerriAI/litellm/issues/18922
        """
        message = {
            "role": "assistant",
            "content": "",
            "thinking": "Let me analyze this request and call the weather function...",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")

        # CRITICAL: Both should be extracted
        assert result.reasoning_content is not None
        assert "analyze this request" in result.reasoning_content

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"

        # Tool calls present → finish_reason should be "tool_calls"
        assert result.finish_reason == "tool_calls"

    def test_content_and_thinking_and_tool_calls(self):
        """Test all three fields present simultaneously."""
        message = {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "thinking": "User wants weather info...",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")

        assert result.content == "I'll check the weather for you."
        assert result.reasoning_content == "User wants weather info..."
        assert result.tool_calls is not None
        assert result.finish_reason == "tool_calls"

    def test_thinking_blocks_and_tool_calls(self):
        """Test Anthropic-style thinking_blocks with tool_calls."""
        message = {
            "role": "assistant",
            "content": "",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "Analyzing the request..."}
            ],
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="anthropic")

        assert result.thinking_blocks is not None
        assert result.reasoning_content is not None
        assert result.tool_calls is not None
        assert result.finish_reason == "tool_calls"


class TestFinishReasonDetermination:
    """Tests for finish_reason logic."""

    def test_tool_calls_sets_finish_reason(self):
        """Test that presence of tool_calls sets finish_reason to 'tool_calls'."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "test", "arguments": {}}}
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.finish_reason == "tool_calls"

    def test_finish_reason_from_message(self):
        """Test that message's finish_reason is respected when no tool_calls."""
        message = {
            "role": "assistant",
            "content": "Done",
            "finish_reason": "length",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.finish_reason == "length"

    def test_done_reason_mapping(self):
        """Test that done_reason (Ollama) is mapped correctly."""
        message = {
            "role": "assistant",
            "content": "Done",
            "done_reason": "stop",
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.finish_reason == "stop"

    def test_end_turn_maps_to_stop(self):
        """Test that 'end_turn' maps to 'stop'."""
        message = {
            "role": "assistant",
            "content": "Hello",
            "finish_reason": "end_turn",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.finish_reason == "stop"

    def test_tool_use_maps_to_tool_calls(self):
        """Test that 'tool_use' finish_reason maps to 'tool_calls'."""
        message = {
            "role": "assistant",
            "content": "",
            "finish_reason": "tool_use",
        }
        result = ResponseFieldExtractor.extract_all(message)
        # Even without actual tool_calls, the finish_reason mapping works
        assert result.finish_reason == "tool_calls"


class TestEdgeCases:
    """Edge case tests."""

    def test_arguments_already_string(self):
        """Test that string arguments pass through unchanged."""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "test",
                        "arguments": '{"already": "string"}',
                    }
                }
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.tool_calls[0]["function"]["arguments"] == '{"already": "string"}'

    def test_empty_arguments_dict(self):
        """Test that empty arguments dict becomes empty JSON object string."""
        message = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "no_args", "arguments": {}}}],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.tool_calls[0]["function"]["arguments"] == "{}"

    def test_missing_arguments(self):
        """Test that missing arguments defaults to empty object."""
        message = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "no_args"}}],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="ollama")
        assert result.tool_calls[0]["function"]["arguments"] == "{}"

    def test_content_list_format(self):
        """Test extraction from Anthropic-style content list."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ],
        }
        result = ResponseFieldExtractor.extract_all(message, provider="anthropic")
        assert result.content == "Hello world!"

    def test_provider_specific_fields_preserved(self):
        """Test that unknown fields are captured in provider_specific_fields."""
        message = {
            "role": "assistant",
            "content": "Hello",
            "custom_field": "some_value",
            "another_field": 123,
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert result.provider_specific_fields["custom_field"] == "some_value"
        assert result.provider_specific_fields["another_field"] == 123

    def test_think_tags_multiline(self):
        """Test <think> tag extraction with multiline content."""
        message = {
            "role": "assistant",
            "content": """<think>
Step 1: Consider the question
Step 2: Formulate response
</think>Here is my answer.""",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert "Step 1" in result.reasoning_content
        assert "Step 2" in result.reasoning_content
        assert result.content == "Here is my answer."

    def test_multiple_think_blocks(self):
        """Test extraction of multiple <think> blocks in content."""
        message = {
            "role": "assistant",
            "content": "<think>First thought</think>Let me also <think>reconsider this</think>Here is the answer.",
        }
        result = ResponseFieldExtractor.extract_all(message)
        # Both thinking blocks should be extracted and concatenated
        assert "First thought" in result.reasoning_content
        assert "reconsider this" in result.reasoning_content
        # Content should have both think blocks removed
        assert result.content == "Let me also Here is the answer."
        assert "<think>" not in result.content

    def test_multiple_think_blocks_with_newlines(self):
        """Test multiple <think> blocks separated by content with newlines."""
        message = {
            "role": "assistant",
            "content": """<think>Initial analysis</think>
Some text here.
<think>Follow-up thought</think>
Final answer.""",
        }
        result = ResponseFieldExtractor.extract_all(message)
        assert "Initial analysis" in result.reasoning_content
        assert "Follow-up thought" in result.reasoning_content
        assert "Some text here" in result.content
        assert "Final answer" in result.content
        assert "<think>" not in result.content
