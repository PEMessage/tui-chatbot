"""TUI Chatbot - A modern terminal UI chatbot with streaming output and TPS statistics."""

__version__ = "0.1.0"

# Config 模块公共 API
from .config import (
    ConfigManager,
    UserConfig,
    get_config_manager,
)

# Agent 模块公共 API
from .agent import (
    # Types
    AgentEvent,
    AgentEventType,
    AgentLoopConfig,
    AgentMessage,
    AssistantMessage,
    ChatResult,
    TextContent,
    ToolCallContent,
    ToolCallMessage,
    ToolExecutionMode,
    ToolResultMessage,
    UserMessage,
    # Tool
    Tool,
    ToolParameters,
    ToolRegistry,
    ToolResult,
    GetCurrentTimeTool,
    GetCurrentTimeParams,
    create_default_tool_registry,
    # Loop
    agent_loop,
    AgentEventSink,
)

# Provider 模块公共 API
from .provider import (
    # Base
    Provider,
    ProviderConfig,
    # Registry
    ProviderRegistry,
    LazyProvider,
    ProviderLoader,
    # OpenAI
    OpenAIProvider,
    OpenAIProviderConfig,
    # Utils
    create_provider_from_env,
    register_default_providers,
)

__all__ = [
    "__version__",
    # Config
    "ConfigManager",
    "UserConfig",
    "get_config_manager",
    # Agent Types
    "AgentEvent",
    "AgentEventType",
    "AgentLoopConfig",
    "AgentMessage",
    "AssistantMessage",
    "ChatResult",
    "TextContent",
    "ToolCallContent",
    "ToolCallMessage",
    "ToolExecutionMode",
    "ToolResultMessage",
    "UserMessage",
    # Tool
    "Tool",
    "ToolParameters",
    "ToolRegistry",
    "ToolResult",
    "GetCurrentTimeTool",
    "GetCurrentTimeParams",
    "create_default_tool_registry",
    # Loop
    "agent_loop",
    "AgentEventSink",
    # Provider Base
    "Provider",
    "ProviderConfig",
    # Provider Registry
    "ProviderRegistry",
    "LazyProvider",
    "ProviderLoader",
    # OpenAI Provider
    "OpenAIProvider",
    "OpenAIProviderConfig",
    # Provider Utils
    "create_provider_from_env",
    "register_default_providers",
]
