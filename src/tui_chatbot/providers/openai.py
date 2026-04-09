"""OpenAI Provider implementation.

Implements the Provider interface for OpenAI-compatible APIs.
Supports streaming with full event protocol, reasoning content detection,
and graceful error handling.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .base import Provider, ProviderConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ..event_stream import AssistantMessageEventStream
from ..events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ..types import (
    AssistantMessage,
    Content,
    Message,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  OpenAI Provider Configuration                               │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class OpenAIProviderConfig(ProviderConfig):
    """OpenAI provider configuration.

    Attributes:
        model: Default model to use
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 - 2.0)
        presence_penalty: Presence penalty (-2.0 - 2.0)
        reasoning_effort: Reasoning effort for o1/o3 models
    """

    model: str = "gpt-4o-mini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    reasoning_effort: Optional[str] = None


# ╭────────────────────────────────────────────────────────────╮
# │  OpenAI Provider Implementation                              │
# ╰────────────────────────────────────────────────────────────╯


class OpenAIProvider(Provider):
    """OpenAI API provider with full event protocol support.

    Features:
    - Streaming chat completions with detailed events
    - Reasoning content detection (o1, o3, Doubao, etc.)
    - Tool call support
    - Usage tracking
    - Abort signal support
    - Compatible with any OpenAI-compatible API

    Example:
        config = OpenAIProviderConfig(api_key="sk-...")
        provider = OpenAIProvider(config)

        stream = await provider.stream(messages, "gpt-4")
        async for event in stream:
            if event.type == "text_delta":
                print(event.delta, end="")

        result = await stream.result()
    """

    def __init__(
        self,
        client: Optional["AsyncOpenAI"] = None,
        config: Optional[OpenAIProviderConfig] = None,
    ):
        """Initialize OpenAI provider.

        Args:
            client: OpenAI client instance (created from config if None)
            config: Provider configuration
        """
        self._config = config or OpenAIProviderConfig()
        self._client = client
        self._client_owned = client is None

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    def _get_client(self) -> "AsyncOpenAI":
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._config.api_key or "",
                base_url=self._config.base_url,
            )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal message types to OpenAI format.

        Args:
            messages: Internal message objects

        Returns:
            OpenAI API compatible message dictionaries
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    result.append({"role": "user", "content": msg.content})
                else:
                    # Handle content blocks
                    content_blocks = []
                    for content in msg.content:
                        if isinstance(content, TextContent):
                            content_blocks.append(
                                {"type": "text", "text": content.text}
                            )
                    result.append({"role": "user", "content": content_blocks})

            elif isinstance(msg, AssistantMessage):
                content_text = ""
                for content in msg.content:
                    if isinstance(content, TextContent):
                        content_text += content.text
                    elif isinstance(content, ToolCall):
                        # Tool calls are handled separately
                        pass
                result.append({"role": "assistant", "content": content_text})

            elif isinstance(msg, ToolResultMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.toolCallId,
                        "content": msg.content,
                    }
                )

        return result

    async def stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs,
    ) -> AssistantMessageEventStream:
        """Stream chat completions with full event protocol.

        Event flow:
            start -> text_start -> text_delta* -> text_end
                  -> thinking_start -> thinking_delta* -> thinking_end
                  -> toolcall_start -> toolcall_delta* -> toolcall_end
                  -> done | error

        Args:
            messages: Conversation messages
            model: Model identifier
            **kwargs: Additional parameters (temperature, max_tokens, signal, etc.)

        Returns:
            AssistantMessageEventStream with full event protocol
        """
        from ..core.abort_controller import AbortSignal

        stream = AssistantMessageEventStream()
        signal: Optional[AbortSignal] = kwargs.get("signal")

        # Extract parameters
        temperature = kwargs.get("temperature", self._config.temperature)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)
        tools = kwargs.get("tools")

        async def _stream() -> None:
            try:
                # Check for abort before starting
                if signal and signal.aborted:
                    error_msg = AssistantMessage(
                        role="assistant",
                        content=[TextContent(text="")],
                        stopReason=StopReason.ABORTED,
                        errorMessage=f"Aborted: {signal.reason}",
                    )
                    stream.push(ErrorEvent(reason=StopReason.ABORTED, error=error_msg))
                    return

                # Build request parameters
                params: dict[str, Any] = {
                    "model": model,
                    "messages": self._convert_messages(messages),
                    "stream": True,
                }

                if temperature is not None:
                    params["temperature"] = temperature
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if tools:
                    params["tools"] = tools

                # Add reasoning effort for o1/o3 models if configured
                if self._config.reasoning_effort and "o1" in model or "o3" in model:
                    params["reasoning_effort"] = self._config.reasoning_effort

                # Send start event
                stream.push(StartEvent())

                # Track state during streaming
                content_index = 0
                current_content: list[Content] = []
                current_text = ""
                current_thinking = ""
                in_text_block = False
                in_thinking_block = False
                in_tool_call = False

                tool_calls: list[dict[str, Any]] = []
                usage_stats = Usage()
                finish_reason: Optional[str] = None

                # Start the API stream
                client = self._get_client()
                api_stream = await client.chat.completions.create(**params)

                async for chunk in api_stream:
                    # Check abort signal
                    if signal and signal.aborted:
                        raise asyncio.CancelledError(f"Aborted: {signal.reason}")

                    if not chunk.choices:
                        # Usage info is sometimes in the final chunk
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_stats = self._parse_usage(chunk.usage)
                        continue

                    delta = chunk.choices[0].delta
                    if not delta:
                        continue

                    # Track finish reason
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                    # Handle reasoning_content (Doubao/Ark/Volces, o1, o3 models)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not in_thinking_block:
                            # Start thinking block
                            in_thinking_block = True
                            current_thinking = ""
                            stream.push(
                                ThinkingStartEvent(
                                    content_index=content_index,
                                    partial=AssistantMessage(
                                        content=current_content.copy(),
                                    ),
                                )
                            )

                        current_thinking += reasoning
                        stream.push(
                            ThinkingDeltaEvent(
                                content_index=content_index,
                                delta=reasoning,
                                partial=AssistantMessage(
                                    content=current_content.copy(),
                                ),
                            )
                        )

                    # Handle regular content
                    content = getattr(delta, "content", None)
                    if content:
                        if not in_text_block:
                            # End thinking block if active
                            if in_thinking_block:
                                in_thinking_block = False
                                thinking = ThinkingContent(thinking=current_thinking)
                                current_content.append(thinking)
                                stream.push(
                                    ThinkingEndEvent(
                                        content_index=content_index,
                                        content=thinking,
                                        partial=AssistantMessage(
                                            content=current_content.copy(),
                                        ),
                                    )
                                )
                                content_index += 1
                                current_thinking = ""

                            # Start text block
                            in_text_block = True
                            current_text = ""
                            stream.push(
                                TextStartEvent(
                                    content_index=content_index,
                                    partial=AssistantMessage(
                                        content=current_content.copy(),
                                    ),
                                )
                            )

                        current_text += content
                        stream.push(
                            TextDeltaEvent(
                                content_index=content_index,
                                delta=content,
                                partial=AssistantMessage(
                                    content=current_content.copy(),
                                ),
                            )
                        )

                    # Handle tool calls
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        if not in_tool_call:
                            in_tool_call = True
                            stream.push(ToolCallStartEvent(content_index=content_index))

                        for tc in delta_tool_calls:
                            tc_index = tc.index
                            # Ensure tool_calls list is long enough
                            while len(tool_calls) <= tc_index:
                                tool_calls.append(
                                    {"id": "", "name": "", "arguments": ""}
                                )

                            if tc.id:
                                tool_calls[tc_index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[tc_index]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[tc_index]["arguments"] += (
                                        tc.function.arguments
                                    )

                            stream.push(
                                ToolCallDeltaEvent(
                                    content_index=content_index,
                                    delta=tc.function.arguments if tc.function else "",
                                )
                            )

                # End any active blocks
                if in_thinking_block:
                    thinking = ThinkingContent(thinking=current_thinking)
                    current_content.append(thinking)
                    stream.push(
                        ThinkingEndEvent(
                            content_index=content_index,
                            content=thinking,
                            partial=AssistantMessage(
                                content=current_content.copy(),
                            ),
                        )
                    )
                    content_index += 1

                if in_text_block:
                    text = TextContent(text=current_text)
                    current_content.append(text)
                    stream.push(
                        TextEndEvent(
                            content_index=content_index,
                            content=text,
                            partial=AssistantMessage(
                                content=current_content.copy(),
                            ),
                        )
                    )
                    content_index += 1

                if in_tool_call:
                    # Emit tool call end events
                    for tc in tool_calls:
                        if tc.get("name"):
                            try:
                                args = json.loads(tc.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}

                            tool_call = ToolCall(
                                id=tc.get("id", ""),
                                name=tc["name"],
                                arguments=args,
                            )
                            current_content.append(tool_call)
                            stream.push(
                                ToolCallEndEvent(
                                    content_index=content_index,
                                    tool_call=tool_call,
                                    partial=AssistantMessage(
                                        content=current_content.copy(),
                                    ),
                                )
                            )
                            content_index += 1

                # Build final message
                stop_reason = self._map_finish_reason(finish_reason)
                final_message = AssistantMessage(
                    role="assistant",
                    content=current_content,
                    provider=self.name,
                    model=model,
                    usage=usage_stats,
                    stopReason=stop_reason,
                )

                # Emit done event
                stream.push(
                    DoneEvent(
                        reason=stop_reason,
                        message=final_message,
                    )
                )

            except asyncio.CancelledError:
                # Build partial message with what we have
                partial_content: list[Content] = []
                if in_thinking_block and current_thinking:
                    partial_content.append(ThinkingContent(thinking=current_thinking))
                if in_text_block and current_text:
                    partial_content.append(TextContent(text=current_text))

                partial_message = AssistantMessage(
                    role="assistant",
                    content=partial_content,
                    provider=self.name,
                    model=model,
                    stopReason=StopReason.ABORTED,
                    errorMessage="Stream was aborted",
                )
                stream.push(
                    ErrorEvent(reason=StopReason.ABORTED, error=partial_message)
                )
                raise

            except Exception as e:
                # Handle errors
                error_message = AssistantMessage(
                    role="assistant",
                    content=[TextContent(text=f"Error: {str(e)}")],
                    provider=self.name,
                    model=model,
                    stopReason=StopReason.ERROR,
                    errorMessage=str(e),
                )
                stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_message))

        # Start streaming task
        asyncio.create_task(_stream())
        return stream

    def _parse_usage(self, usage: Any) -> Usage:
        """Parse usage information from API response.

        Args:
            usage: OpenAI usage object

        Returns:
            Internal Usage dataclass
        """
        return Usage(
            inputTokens=getattr(usage, "prompt_tokens", 0),
            outputTokens=getattr(usage, "completion_tokens", 0),
            totalTokens=getattr(usage, "total_tokens", 0),
        )

    def _map_finish_reason(self, finish_reason: Optional[str]) -> StopReason:
        """Map OpenAI finish reason to internal StopReason.

        Args:
            finish_reason: OpenAI finish reason string

        Returns:
            Internal StopReason enum value
        """
        if not finish_reason:
            return StopReason.END_TURN

        mapping = {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "content_filter": StopReason.ERROR,
        }
        return mapping.get(finish_reason, StopReason.END_TURN)

    async def list_models(self) -> list[str]:
        """List available models from OpenAI.

        Returns:
            List of model identifiers
        """
        try:
            client = self._get_client()
            models = await client.models.list()
            return [m.id for m in models.data]
        except Exception:
            # Return default models if API call fails
            return [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "o1",
                "o1-mini",
                "o3-mini",
            ]
