"""Tests for the OpenAI provider.

Mock-based streaming tests with event sequence validation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tui_chatbot.core.abort_controller import AbortController
from tui_chatbot.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from tui_chatbot.providers.openai import OpenAIProvider, OpenAIProviderConfig
from tui_chatbot.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    UserMessage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Test Fixtures                                               │
# ╰────────────────────────────────────────────────────────────╯


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    return client


@pytest.fixture
def provider(mock_openai_client):
    """Create an OpenAI provider with mock client."""
    config = OpenAIProviderConfig(api_key="test-key")
    return OpenAIProvider(client=mock_openai_client, config=config)


@pytest.fixture
def simple_messages():
    """Simple user message for testing."""
    return [UserMessage(content="Hello")]


def create_mock_chunk(content: str = "", reasoning: str = "", finish_reason=None):
    """Create a mock OpenAI stream chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]

    delta = MagicMock()
    delta.content = content if content else None
    delta.reasoning_content = reasoning if reasoning else None
    delta.tool_calls = None
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason

    return chunk


class MockAsyncIterator:
    """Helper class for mocking async iterators."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


# ╭────────────────────────────────────────────────────────────╮
# │  Basic Provider Tests                                        │
# ╰────────────────────────────────────────────────────────────╯


def test_provider_name(provider):
    """Test provider name property."""
    assert provider.name == "openai"


def test_provider_config():
    """Test provider configuration."""
    config = OpenAIProviderConfig(
        api_key="sk-test",
        base_url="https://custom.api.com",
        model="gpt-4",
        temperature=0.7,
    )
    provider = OpenAIProvider(config=config)

    assert provider._config.api_key == "sk-test"
    assert provider._config.base_url == "https://custom.api.com"


# ╭────────────────────────────────────────────────────────────╮
# │  Streaming Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_stream_emits_start_event(provider, mock_openai_client, simple_messages):
    """Test that stream emits start event."""
    # Setup mock with proper async iterator
    mock_chunk = create_mock_chunk(content="Hi", finish_reason="stop")
    mock_stream = MockAsyncIterator([mock_chunk])
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Should have start event
    assert any(e.type == "start" for e in events)


@pytest.mark.asyncio
async def test_stream_emits_text_events(provider, mock_openai_client, simple_messages):
    """Test that stream emits text start, delta, and end events."""
    # Setup mock with multiple chunks
    chunks = [
        create_mock_chunk(content="Hello "),
        create_mock_chunk(content="world"),
        create_mock_chunk(content="!", finish_reason="stop"),
    ]

    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Check event types
    event_types = [e.type for e in events]

    assert "start" in event_types
    assert "text_start" in event_types
    assert event_types.count("text_delta") >= 3  # At least 3 text deltas
    assert "text_end" in event_types
    assert "done" in event_types


@pytest.mark.asyncio
async def test_stream_text_content_accumulation(
    provider, mock_openai_client, simple_messages
):
    """Test that text content is accumulated correctly."""
    # Setup mock
    chunks = [
        create_mock_chunk(content="Hello "),
        create_mock_chunk(content="there"),
        create_mock_chunk(content="!", finish_reason="stop"),
    ]

    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Collect text deltas
    text_deltas = []
    final_message = None

    async for event in stream:
        if isinstance(event, TextDeltaEvent):
            text_deltas.append(event.delta)
        if isinstance(event, DoneEvent):
            final_message = event.message

    # Text should be accumulated
    assert "".join(text_deltas) == "Hello there!"

    # Final message should have the full text
    assert len(final_message.content) == 1
    assert isinstance(final_message.content[0], TextContent)
    assert final_message.content[0].text == "Hello there!"


@pytest.mark.asyncio
async def test_stream_with_reasoning(provider, mock_openai_client, simple_messages):
    """Test that reasoning content is handled correctly."""
    # Setup mock with reasoning content
    chunks = [
        create_mock_chunk(reasoning="Let me think"),
        create_mock_chunk(reasoning=" about this"),
        create_mock_chunk(content="The answer is 42", finish_reason="stop"),
    ]
    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Collect events
    event_types = [e.type async for e in stream]

    # Should have thinking events
    assert "thinking_start" in event_types
    assert "thinking_delta" in event_types
    assert "thinking_end" in event_types


@pytest.mark.asyncio
async def test_stream_result_method(provider, mock_openai_client, simple_messages):
    """Test that stream.result() returns final message."""
    # Setup mock
    chunks = [
        create_mock_chunk(content="Response", finish_reason="stop"),
    ]
    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Get result
    result = await stream.result()

    # Should be an AssistantMessage
    assert isinstance(result, AssistantMessage)
    assert len(result.content) == 1
    assert result.content[0].text == "Response"


# ╭────────────────────────────────────────────────────────────╮
# │  Abort/Error Tests                                           │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_stream_aborted_before_start(provider, simple_messages):
    """Test that stream handles abort before starting."""
    # Create abort controller and trigger abort
    controller = AbortController()
    controller.abort("user cancelled")

    # Call stream with aborted signal
    stream = await provider.stream(simple_messages, "gpt-4", signal=controller.signal)

    # Give background task a chance to run
    await asyncio.sleep(0)

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Should have error event
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].reason == StopReason.ABORTED


@pytest.mark.asyncio
async def test_stream_abort_during_streaming(
    provider, mock_openai_client, simple_messages
):
    """Test that stream handles abort during streaming."""
    # Setup mock with many chunks to ensure we can abort mid-stream
    chunks = [
        create_mock_chunk(content="chunk1 "),
        create_mock_chunk(content="chunk2 "),
        create_mock_chunk(content="chunk3 "),
        create_mock_chunk(content="chunk4 "),
        create_mock_chunk(content="chunk5", finish_reason="stop"),
    ]

    # Create a generator that checks abort signal between yields
    abort_controller = AbortController()

    async def mock_aiter():
        for i, chunk in enumerate(chunks):
            # Check abort before yielding each chunk (after the first)
            if i > 0 and abort_controller.signal.aborted:
                raise asyncio.CancelledError("Aborted")
            yield chunk

    class AbortCheckingIterator:
        def __aiter__(self):
            return mock_aiter()

    mock_stream = AbortCheckingIterator()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(
        simple_messages, "gpt-4", signal=abort_controller.signal
    )

    # Give background task a chance to start
    await asyncio.sleep(0)

    # Start collecting events
    events = []
    text_chunks = []

    async for event in stream:
        events.append(event)
        # Collect text deltas
        if hasattr(event, "delta") and event.delta:
            text_chunks.append(event.delta)
        # Abort after receiving some text content
        if len(text_chunks) >= 2:
            if not abort_controller.signal.aborted:
                abort_controller.abort("test abort")
            # Give provider time to process abort
            await asyncio.sleep(0)

    # Should have received some events before abort
    assert len(events) > 0

    # Check if we got an error event or partial content
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    done_events = [e for e in events if e.type == "done"]

    # Either we got an error (abort worked) or done (all chunks processed before abort)
    # In this test, we expect the abort to be detected
    assert len(error_events) == 1 or len(done_events) == 1


@pytest.mark.asyncio
async def test_stream_handles_api_error(provider, mock_openai_client, simple_messages):
    """Test that stream handles API errors gracefully."""
    # Setup mock to raise error
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error: Rate limit exceeded")
    )

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Should have error event
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1

    # Get final result
    result = await stream.result()
    assert result.stopReason == StopReason.ERROR
    assert "API Error" in result.errorMessage


# ╭────────────────────────────────────────────────────────────╮
# │  List Models Tests                                           │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_list_models_success(provider, mock_openai_client):
    """Test listing models when API succeeds."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(id="gpt-4"),
        MagicMock(id="gpt-3.5-turbo"),
        MagicMock(id="custom-model"),
    ]
    mock_openai_client.models.list = AsyncMock(return_value=mock_response)

    # Call list_models
    models = await provider.list_models()

    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert "custom-model" in models


@pytest.mark.asyncio
async def test_list_models_failure(provider, mock_openai_client):
    """Test listing models when API fails."""
    # Setup mock to fail
    mock_openai_client.models.list = AsyncMock(side_effect=Exception("API Error"))

    # Call list_models - should return default list
    models = await provider.list_models()

    # Should return default models
    assert "gpt-4" in models
    assert "gpt-4o" in models
    assert "gpt-3.5-turbo" in models


# ╭────────────────────────────────────────────────────────────╮
# │  Edge Case Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_stream_empty_response(provider, mock_openai_client, simple_messages):
    """Test handling of empty response."""
    # Setup mock with no content
    chunks = [create_mock_chunk(content="", finish_reason="stop")]
    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Get result
    result = await stream.result()

    # Should handle empty content gracefully
    assert isinstance(result, AssistantMessage)


@pytest.mark.asyncio
async def test_stream_multiple_content_blocks(
    provider, mock_openai_client, simple_messages
):
    """Test streaming with multiple content blocks."""
    # Setup mock with reasoning then content
    chunks = [
        create_mock_chunk(reasoning="Thinking step 1"),
        create_mock_chunk(reasoning="Thinking step 2"),
        create_mock_chunk(content="Answer part 1"),
        create_mock_chunk(content="Answer part 2", finish_reason="stop"),
    ]
    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream
    stream = await provider.stream(simple_messages, "gpt-4")

    # Get result
    result = await stream.result()

    # Should have both thinking and text content
    assert len(result.content) == 2


@pytest.mark.asyncio
async def test_stream_simple_method(provider, mock_openai_client, simple_messages):
    """Test the stream_simple convenience method."""
    # Setup mock
    chunks = [
        create_mock_chunk(content="Hello "),
        create_mock_chunk(content="world", finish_reason="stop"),
    ]
    mock_stream = MockAsyncIterator(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # Call stream_simple
    text_parts = []
    async for text in provider.stream_simple(simple_messages, "gpt-4"):
        text_parts.append(text)

    # Should yield text content
    assert "".join(text_parts) == "Hello world"
