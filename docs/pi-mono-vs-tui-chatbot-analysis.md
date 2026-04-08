# pi-mono Core Runtime vs tui-chatbot 深度对比研究报告

## 1. pi-mono Core Runtime 架构总览

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          @mariozechner/pi-agent-core                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Agent Class                                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐│ │
│  │  │   State     │  │   Queues    │  │  Listeners  │  │  Lifecycle Mgmt ││ │
│  │  │  (mutable)  │  │(steer/follow│  │  (async)    │  │ (abort/idle)   ││ │
│  │  └─────────────┘  │    -up)     │  └─────────────┘  └─────────────────┘│ │
│  │                   └─────────────┘                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Agent Loop (agent-loop.ts)                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐│ │
│  │  │  Outer Loop │  │  Inner Loop │  │ Tool Exec   │  │ Event Emitter   ││ │
│  │  │(follow-ups) │  │(tool/steer) │  │(seq/parallel│  │ (EventStream)   ││ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          @mariozechner/pi-ai                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      API Registry (api-registry.ts)                      │ │
│  │     ┌─────────────┐      ┌─────────────┐      ┌─────────────────────┐ │ │
│  │     │  Provider   │◄────►│   Stream    │◄────►│  AssistantMessage   │ │ │
│  │     │  Registry   │      │  Functions  │      │    EventStream      │ │ │
│  │     └─────────────┘      └─────────────┘      └─────────────────────┘ │ │
│  │            │                                                            │ │
│  │            ▼                                                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Provider Implementations                      │   │ │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │   │ │
│  │  │  │ OpenAI  │ │Anthropic│ │ Google  │ │ Mistral │ │   Bedrock   │ │   │ │
│  │  │  │(2 APIs) │ │(OAuth)  │ │(OAuth)  │ │         │ │  (Node only)│ │   │ │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘ │   │ │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │   │ │
│  │  │  │  xAI    │ │  Groq   │ │Cerebras │ │OpenRouter│ │ GitHub Cop │ │   │ │
│  │  │  │         │ │         │ │         │ │          │ │  (OAuth)   │ │   │ │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 实现 |
|------|------|
| **Protocol-First** | 所有 LLM 输出统一为标准化的 `AssistantMessageEvent` 流 |
| **Provider-Agnostic** | 统一的 `Model<TApi>` 类型，15+ 提供商共享相同接口 |
| **Type Safety** | TypeBox Schema 用于工具定义，运行时 AJV 验证 |
| **Lazy Loading** | Provider 模块按需加载，减少启动开销 |
| **Event-Driven** | 完整的生命周期事件系统（10+ 事件类型） |
| **Extensible** | 自定义消息类型通过声明合并（Declaration Merging）扩展 |

---

## 2. 关键设计模式分析

### 2.1 EventStream 模式（流式处理核心）

```typescript
// packages/ai/src/utils/event-stream.ts
export class EventStream<T, R = T> implements AsyncIterable<T> {
    private queue: T[] = [];
    private waiting: ((value: IteratorResult<T>) => void)[] = [];
    private done = false;
    private finalResultPromise: Promise<R>;

    push(event: T): void {
        if (this.done) return;
        if (this.isComplete(event)) {
            this.done = true;
            this.resolveFinalResult(this.extractResult(event));
        }
        const waiter = this.waiting.shift();
        if (waiter) waiter({ value: event, done: false });
        else this.queue.push(event);
    }

    async *[Symbol.asyncIterator](): AsyncIterator<T> {
        while (true) {
            if (this.queue.length > 0) yield this.queue.shift()!;
            else if (this.done) return;
            else {
                const result = await new Promise<IteratorResult<T>>(
                    resolve => this.waiting.push(resolve)
                );
                if (result.done) return;
                yield result.value;
            }
        }
    }
}
```

**优点：**
- 同时支持 `for await...of` 迭代和 `.result()` Promise 获取最终结果
- 生产者-消费者解耦，无需缓冲区大小限制
- 完成检测内置于流中，避免回调地狱

### 2.2 Provider Registry 模式（可扩展架构）

```typescript
// packages/ai/src/api-registry.ts
const apiProviderRegistry = new Map<string, RegisteredApiProvider>();

export function registerApiProvider<TApi extends Api>(
    provider: ApiProvider<TApi>
): void {
    apiProviderRegistry.set(provider.api, {
        provider: {
            api: provider.api,
            stream: wrapStream(provider.api, provider.stream),
            streamSimple: wrapStreamSimple(provider.api, provider.streamSimple),
        }
    });
}

export function getApiProvider(api: Api): ApiProviderInternal | undefined {
    return apiProviderRegistry.get(api)?.provider;
}
```

**优点：**
- 运行时动态注册，支持第三方扩展
- API 标识符与实现解耦，便于测试（Faux Provider）
- 统一的 `stream` / `streamSimple` 入口点

### 2.3 Lazy Provider Loading（性能优化）

```typescript
// packages/ai/src/providers/register-builtins.ts
function createLazyStream<TApi extends Api>(
    loadModule: () => Promise<LazyProviderModule<TApi, any, any>>
): StreamFunction<TApi, any> {
    return (model, context, options) => {
        const outer = new AssistantMessageEventStream();
        loadModule()
            .then(module => forwardStream(outer, module.stream(model, context, options)))
            .catch(error => outer.push({ type: "error", ... }));
        return outer;
    };
}

// 使用 Promise 缓存避免重复加载
let anthropicProviderModulePromise: Promise<...> | undefined;
function loadAnthropicProviderModule() {
    anthropicProviderModulePromise ||= import("./anthropic.js").then(...);
    return anthropicProviderModulePromise;
}
```

**优点：**
- 启动时间优化，只加载实际使用的提供商
- 模块级缓存确保单例
- 错误处理在流中传播，不抛出异常

### 2.4 Agent Loop 模式（状态机 + 事件）

```typescript
// packages/agent/src/agent-loop.ts
async function runLoop(
    currentContext: AgentContext,
    newMessages: AgentMessage[],
    config: AgentLoopConfig,
    signal: AbortSignal | undefined,
    emit: AgentEventSink,
    streamFn?: StreamFn
): Promise<void> {
    let firstTurn = true;
    let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

    // Outer loop: 处理 follow-up 消息
    while (true) {
        let hasMoreToolCalls = true;

        // Inner loop: 处理工具调用和 steering 消息
        while (hasMoreToolCalls || pendingMessages.length > 0) {
            if (!firstTurn) await emit({ type: "turn_start" });
            else firstTurn = false;

            // 处理待处理消息
            if (pendingMessages.length > 0) { ... }

            // 流式助手响应
            const message = await streamAssistantResponse(...);
            newMessages.push(message);

            if (message.stopReason === "error" || message.stopReason === "aborted") {
                await emit({ type: "agent_end", messages: newMessages });
                return;
            }

            // 执行工具调用
            const toolCalls = message.content.filter(c => c.type === "toolCall");
            hasMoreToolCalls = toolCalls.length > 0;
            if (hasMoreToolCalls) {
                const toolResults = await executeToolCalls(...);
                for (const result of toolResults) {
                    currentContext.messages.push(result);
                    newMessages.push(result);
                }
            }

            await emit({ type: "turn_end", message, toolResults });
            pendingMessages = (await config.getSteeringMessages?.()) || [];
        }

        // 检查 follow-up 消息
        const followUpMessages = (await config.getFollowUpMessages?.()) || [];
        if (followUpMessages.length > 0) {
            pendingMessages = followUpMessages;
            continue;
        }
        break;
    }
    await emit({ type: "agent_end", messages: newMessages });
}
```

**优点：**
- 双层循环结构清晰：外层处理会话延续，内层处理单轮交互
- Steering 机制允许运行时干预对话流程
- 工具执行支持并行/串行两种模式

### 2.5 Tool Execution 模式（并行/串行可配置）

```typescript
// packages/agent/src/agent-loop.ts
async function executeToolCallsParallel(...): Promise<ToolResultMessage[]> {
    const results: ToolResultMessage[] = [];
    const runnableCalls: PreparedToolCall[] = [];

    // Phase 1: Sequential preflight (validation + beforeToolCall hooks)
    for (const toolCall of toolCalls) {
        const preparation = await prepareToolCall(...);
        if (preparation.kind === "immediate") {
            results.push(await emitToolCallOutcome(...));
        } else {
            runnableCalls.push(preparation);
        }
    }

    // Phase 2: Concurrent execution
    const runningCalls = runnableCalls.map(prepared => ({
        prepared,
        execution: executePreparedToolCall(prepared, signal, emit),
    }));

    // Phase 3: Sequential finalization (afterToolCall hooks, ordered by assistant source)
    for (const running of runningCalls) {
        const executed = await running.execution;
        results.push(await finalizeExecutedToolCall(...));
    }
    return results;
}
```

**优点：**
- 三阶段执行：Preflight（串行验证）→ Execution（并行执行）→ Finalization（串行后处理）
- 结果保持原始顺序，符合 LLM 期望
- Hooks (beforeToolCall/afterToolCall) 提供拦截点

---

## 3. 与 tui-chatbot 的对比矩阵

### 3.1 架构设计对比

| 维度 | pi-mono | tui-chatbot | 差距分析 |
|------|---------|-------------|----------|
| **事件系统** | 完整的事件生命周期（agent_start/turn_start/message_start/message_update/message_end/turn_end/agent_end + tool_execution_*） | 简单的事件枚举（REASONING_TOKEN/CONTENT_TOKEN/STATS/DONE/ERROR） | tui-chatbot 事件粒度较粗，缺乏工具执行生命周期 |
| **状态管理** | MutableAgentState + 访问器属性，自动数组拷贝 | 简单的 msgs 列表，手动 trim_history | tui-chatbot 状态管理较简单，缺乏 reactive 更新 |
| **模块化** | 清晰的 packages/ai → packages/agent 分层 | Daemon + Frontend + Shell 三层 | 架构分层相似，但 pi-mono 抽象更深 |
| **流式协议** | EventStream 类，支持 async iteration + Promise result | 原生 AsyncGenerator | pi-mono 更灵活，同时支持迭代和一次性结果获取 |
| **上下文转换** | convertToLlm + transformContext 双重转换管道 | 直接传递消息到 OpenAI 客户端 | tui-chatbot 缺乏转换抽象，难以扩展自定义消息类型 |
| **工具架构** | AgentTool 接口（含 label, prepareArguments, execute） | 无内置工具架构 | tui-chatbot 完全缺乏工具调用能力 |

### 3.2 API 设计对比

| 维度 | pi-mono | tui-chatbot | 差距分析 |
|------|---------|-------------|----------|
| **流式 API** | stream/streamSimple + complete/completeSimple 双重接口 | 单一的 chat() AsyncGenerator | pi-mono 提供高级别和精细控制两种接口 |
| **错误处理** | 流内 error 事件，stopReason 标记（error/aborted） | try/catch + ERROR 事件 | pi-mono 的错误处理更标准化，支持 abort 恢复 |
| **取消机制** | AbortController 全程传递 | asyncio.CancelledError 捕获 | pi-mono 更完整的取消信号传播 |
| **跨提供商** | 15+ 提供商，统一接口 | 仅 OpenAI | tui-chatbot 需要大量工作支持多提供商 |
| **Steering** | 运行时消息队列注入 | 不支持 | tui-chatbot 无法在运行时干预对话 |
| **Follow-up** | 会话结束后自动延续队列 | 不支持 | tui-chatbot 需要手动重新启动对话 |

### 3.3 可扩展性对比

| 维度 | pi-mono | tui-chatbot | 差距分析 |
|------|---------|-------------|----------|
| **添加提供商** | 注册表模式：实现 stream/streamSimple → registerApiProvider | 需要修改 Daemon 类 | pi-mono 插件化，tui-chatbot 需要侵入式修改 |
| **自定义消息** | 声明合并扩展 CustomAgentMessages 接口 | 需要修改 EventType 和 Daemon | pi-mono 类型安全，tui-chatbot 需要手动维护 |
| **工具扩展** | AgentTool<TParameters, TDetails> 泛型接口 | 无工具支持 | tui-chatbot 完全缺失 |
| **Hook 系统** | beforeToolCall / afterToolCall | 无 | tui-chatbot 缺乏拦截机制 |

### 3.4 类型安全与异步模型

| 维度 | pi-mono (TypeScript) | tui-chatbot (Python) | 对比分析 |
|------|---------------------|---------------------|----------|
| **Schema 定义** | TypeBox（JSON Schema 生成 + 运行时验证） | dataclass（仅类型提示） | TypeBox 提供运行时验证，dataclass 需要额外验证库 |
| **模型类型** | Model<TApi> 泛型，API 与选项类型关联 | 简单的字符串 model 名称 | pi-mono 类型更精确，编译时捕获错误 |
| **异步模式** | async/await + EventStream + AbortSignal | async/await + AsyncGenerator | 模式相似，pi-mono 的 EventStream 更灵活 |
| **事件类型** | 联合类型（Discriminated Union） | Enum + match | TypeScript 的联合类型更精确，Python 3.10+ match 类似 |

---

## 4. 改进建议（基于 pi-mono 设计）

### 4.1 高优先级（High）

#### S1: 引入分层事件系统
**现状问题：** tui-chatbot 只有 5 种事件类型，无法支持工具调用和复杂交互。

**改进方案：**
```python
# 建议的事件层级
class AgentEventType(Enum):
    # 生命周期
    AGENT_START = auto()
    AGENT_END = auto()
    
    # Turn 生命周期
    TURN_START = auto()
    TURN_END = auto()
    
    # 消息生命周期
    MESSAGE_START = auto()
    MESSAGE_UPDATE = auto()  # 流式更新
    MESSAGE_END = auto()
    
    # 工具生命周期
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_UPDATE = auto()  # 工具流式更新
    TOOL_EXECUTION_END = auto()
```

**预期收益：**
- 支持工具调用可视化
- 更精细的 UI 更新控制
- 可拦截和记录完整交互流程

**实现难度：** 中等（需要重构 Daemon 类）

---

#### S2: 实现 Provider Registry 模式
**现状问题：** 仅支持 OpenAI，切换模型需要修改代码。

**改进方案：**
```python
# 建议的注册表模式
class ProviderRegistry:
    _providers: Dict[str, Provider] = {}
    
    @classmethod
    def register(cls, api: str, provider: Provider) -> None:
        cls._providers[api] = provider
    
    @classmethod
    def get(cls, api: str) -> Optional[Provider]:
        return cls._providers.get(api)

# 延迟加载示例
async def stream(api: str, model: str, context: Context, options: StreamOptions) -> EventStream:
    provider = ProviderRegistry.get(api)
    if not provider:
        raise ValueError(f"Unknown API: {api}")
    return await provider.stream(model, context, options)
```

**预期收益：**
- 运行时动态添加提供商
- 统一的接口支持 15+ 提供商
- 便于测试（Mock Provider）

**实现难度：** 中等

---

#### S3: 添加工具调用架构
**现状问题：** 完全缺乏工具支持，无法构建 Agent。

**改进方案：**
```python
from dataclasses import dataclass
from typing import Callable, Awaitable, Any
from pydantic import BaseModel

@dataclass
class Tool:
    name: str
    description: str
    parameters: type[BaseModel]  # Pydantic 替代 TypeBox
    execute: Callable[..., Awaitable[ToolResult]]

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    async def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            raise ToolNotFoundError(name)
        # Pydantic 验证
        validated = tool.parameters(**arguments)
        return await tool.execute(**validated.dict())
```

**预期收益：**
- 支持函数调用，扩展助手能力
- 类型安全的参数验证
- 可复用的工具库

**实现难度：** 高（需要重构核心逻辑）

---

### 4.2 中优先级（Medium）

#### M1: 实现流式结果类（类似 EventStream）
**现状问题：** 只能使用 `async for`，无法同时获取 Promise 风格结果。

**改进方案：**
```python
import asyncio
from typing import AsyncIterator, Generic, TypeVar

T = TypeVar('T')
R = TypeVar('R')

class EventStream(Generic[T, R], AsyncIterator[T]):
    """同时支持 async for 和 await result()"""
    
    def __init__(self):
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._done = asyncio.Event()
        self._result: Optional[R] = None
        self._result_future: asyncio.Future[R] = asyncio.Future()
    
    def push(self, event: T) -> None:
        if self._done.is_set():
            return
        self._queue.put_nowait(event)
    
    def end(self, result: Optional[R] = None) -> None:
        if result is not None:
            self._result_future.set_result(result)
        self._done.set()
    
    async def result(self) -> R:
        return await self._result_future
    
    def __aiter__(self) -> AsyncIterator[T]:
        return self
    
    async def __anext__(self) -> T:
        if self._done.is_set() and self._queue.empty():
            raise StopAsyncIteration
        return await asyncio.wait_for(self._queue.get(), timeout=None)
```

**预期收益：**
- 同时支持迭代和结果获取
- 便于在流结束后获取完整消息
- 与 pi-mono 接口风格一致

**实现难度：** 低

---

#### M2: 添加跨提供商消息转换
**现状问题：** 消息格式与 OpenAI 强耦合。

**改进方案：**
```python
from abc import ABC, abstractmethod

class MessageConverter(ABC):
    @abstractmethod
    def to_provider(self, messages: List[Message]) -> List[dict]:
        """转换为提供商特定格式"""
        pass
    
    @abstractmethod
    def from_provider(self, response: dict) -> AssistantMessage:
        """从提供商响应转换为统一格式"""
        pass

class OpenAIConverter(MessageConverter):
    def to_provider(self, messages: List[Message]) -> List[dict]:
        # OpenAI 格式转换
        return [...]

class AnthropicConverter(MessageConverter):
    def to_provider(self, messages: List[Message]) -> List[dict]:
        # Anthropic 格式转换，处理 thinking 块等
        return [...]
```

**预期收益：**
- 支持多提供商切换
- 统一的内部消息表示
- 便于添加新提供商

**实现难度：** 中等

---

#### M3: 添加 AbortController 模式
**现状问题：** 取消依赖 asyncio.CancelledError，不够精细。

**改进方案：**
```python
import asyncio
from contextlib import asynccontextmanager

class AbortController:
    def __init__(self):
        self._event = asyncio.Event()
        self._reason: Optional[str] = None
    
    @property
    def signal(self) -> 'AbortSignal':
        return AbortSignal(self._event)
    
    def abort(self, reason: str = "aborted") -> None:
        self._reason = reason
        self._event.set()

class AbortSignal:
    def __init__(self, event: asyncio.Event):
        self._event = event
    
    @property
    def aborted(self) -> bool:
        return self._event.is_set()
    
    async def wait(self) -> None:
        await self._event.wait()

# 使用示例
async def chat(self, text: str, signal: Optional[AbortSignal] = None) -> EventStream:
    stream = EventStream()
    
    async def _stream():
        try:
            async for chunk in api_stream:
                if signal and signal.aborted:
                    raise AbortError("Request aborted")
                stream.push(Event(...))
        except Exception as e:
            stream.push(Event(EventType.ERROR, str(e)))
        finally:
            stream.end()
    
    asyncio.create_task(_stream())
    return stream
```

**预期收益：**
- 标准化的取消信号传播
- 支持超时和手动取消
- 与 pi-mono 风格一致

**实现难度：** 低

---

### 4.3 低优先级（Low）

#### L1: 添加 Steering 和 Follow-up 队列
**现状问题：** 无法在运行时干预对话流程。

**改进方案：**
```python
class MessageQueue:
    def __init__(self, mode: QueueMode = QueueMode.ONE_AT_A_TIME):
        self._mode = mode
        self._messages: List[Message] = []
    
    def enqueue(self, message: Message) -> None:
        self._messages.append(message)
    
    def drain(self) -> List[Message]:
        if self._mode == QueueMode.ALL:
            messages = self._messages[:]
            self._messages.clear()
            return messages
        else:
            return [self._messages.pop(0)] if self._messages else []
```

**使用场景：**
- Steering：用户在助手思考时输入新指令
- Follow-up：对话结束后自动执行摘要

**实现难度：** 中等（需要修改 Agent Loop）

---

#### L2: 使用 Pydantic 替代 dataclass（Schema 验证）
**现状问题：** dataclass 只有类型提示，无运行时验证。

**改进方案：**
```python
from pydantic import BaseModel, Field
from typing import Literal

class ToolParameters(BaseModel):
    """工具参数基类，使用 Pydantic 验证"""
    pass

class WeatherToolParams(ToolParameters):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: type[ToolParameters]
    
    def validate_args(self, args: dict) -> ToolParameters:
        return self.parameters(**args)  # Pydantic 自动验证
```

**预期收益：**
- 运行时类型验证
- 自动生成 JSON Schema（用于 LLM）
- 清晰的错误消息

**实现难度：** 低

---

## 5. 可借鉴的代码模式

### 5.1 流式 JSON 解析（工具参数增量解析）

```typescript
// packages/ai/src/utils/json-parse.ts
export function parseStreamingJson(partialJson: string): Record<string, any> | null {
    try {
        return JSON.parse(partialJson);
    } catch {
        // Try to parse incomplete JSON by adding closing brackets
        const attempts = [
            partialJson + '"}',
            partialJson + '}', 
            partialJson + ']}',
            partialJson + '}}',
        ];
        for (const attempt of attempts) {
            try {
                return JSON.parse(attempt);
            } catch {
                continue;
            }
        }
        return null;
    }
}
```

**Python 版本：**
```python
import json
from typing import Any

def parse_streaming_json(partial: str) -> dict[str, Any] | None:
    """尝试解析不完整的 JSON，用于流式工具参数"""
    try:
        return json.loads(partial)
    except json.JSONDecodeError:
        # 尝试补全常见的不完整结构
        attempts = [
            partial + '"}',
            partial + '}',
            partial + ']}',
            partial + '}}',
        ]
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        return None
```

---

### 5.2 双重转换管道（Context → LLM Messages）

```typescript
// packages/agent/src/types.ts - AgentLoopConfig
export interface AgentLoopConfig extends SimpleStreamOptions {
    // 第一层：AgentMessage 级别的转换（上下文窗口管理）
    transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
    
    // 第二层：转换为 LLM 格式（过滤自定义消息）
    convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
}
```

**Python 版本：**
```python
from typing import Callable, Awaitable, Optional
from dataclasses import dataclass

TransformContext = Callable[[list[AgentMessage], Optional[AbortSignal]], Awaitable[list[AgentMessage]]]]
ConvertToLlm = Callable[[list[AgentMessage]], list[Message]]

@dataclass
class AgentConfig:
    transform_context: Optional[TransformContext] = None
    convert_to_llm: ConvertToLlm = lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")]

# 使用
async def process_messages(
    messages: list[AgentMessage],
    config: AgentConfig,
    signal: Optional[AbortSignal] = None
) -> list[Message]:
    # 第一层转换
    if config.transform_context:
        messages = await config.transform_context(messages, signal)
    
    # 第二层转换
    return config.convert_to_llm(messages)
```

---

### 5.3 工具执行三阶段模式

```typescript
// packages/agent/src/agent-loop.ts - executeToolCallsParallel
// Phase 1: Sequential preflight (validation + beforeToolCall hooks)
// Phase 2: Concurrent execution
// Phase 3: Sequential finalization (afterToolCall hooks, in source order)
```

**Python 版本：**
```python
from dataclasses import dataclass
from typing import Awaitable
import asyncio

@dataclass
class PreparedToolCall:
    tool_call: ToolCall
    tool: Tool
    args: dict

@dataclass  
class ExecutedToolCall:
    result: ToolResult
    is_error: bool

async def execute_tools_parallel(
    tool_calls: list[ToolCall],
    tools: list[Tool],
    config: AgentConfig,
    signal: Optional[AbortSignal]
) -> list[ToolResultMessage]:
    # Phase 1: Sequential preflight
    prepared: list[PreparedToolCall] = []
    immediate_results: list[ToolResultMessage] = []
    
    for tc in tool_calls:
        prep = await prepare_tool_call(tc, tools, config, signal)
        if prep.kind == "immediate":
            immediate_results.append(prep.result)
        else:
            prepared.append(prep)
    
    # Phase 2: Concurrent execution
    async def execute_one(p: PreparedToolCall) -> ExecutedToolCall:
        try:
            result = await p.tool.execute(p.tool_call.id, p.args, signal)
            return ExecutedToolCall(result, False)
        except Exception as e:
            return ExecutedToolCall(ToolResult(str(e)), True)
    
    executed = await asyncio.gather(*[execute_one(p) for p in prepared])
    
    # Phase 3: Sequential finalization (maintain order)
    final_results: list[ToolResultMessage] = []
    for prep, exec_result in zip(prepared, executed):
        # Apply afterToolCall hook
        if config.after_tool_call:
            exec_result = await config.after_tool_call(prep, exec_result)
        final_results.append(create_tool_result_message(prep, exec_result))
    
    return immediate_results + final_results
```

---

### 5.4 声明合并扩展模式（Python 替代方案）

TypeScript 使用声明合并扩展类型：
```typescript
declare module "@mariozechner/pi-agent-core" {
  interface CustomAgentMessages {
    notification: NotificationMessage;
  }
}
```

**Python 替代方案（使用 Protocol 和运行时注册）：**
```python
from typing import Protocol, runtime_checkable, TypeVar, Generic
from dataclasses import dataclass

# 基础消息协议
@runtime_checkable
class Message(Protocol):
    role: str
    timestamp: int

# 扩展注册表
class MessageTypeRegistry:
    _types: dict[str, type] = {}
    
    @classmethod
    def register(cls, role: str, message_type: type) -> None:
        cls._types[role] = message_type
    
    @classmethod
    def create(cls, role: str, **data) -> Message:
        msg_type = cls._types.get(role)
        if not msg_type:
            raise ValueError(f"Unknown message type: {role}")
        return msg_type(**data)

# 自定义消息类型
@dataclass
class NotificationMessage:
    role: str = "notification"
    text: str = ""
    timestamp: int = 0

# 注册
MessageTypeRegistry.register("notification", NotificationMessage)

# 使用
msg = MessageTypeRegistry.create("notification", text="Hello", timestamp=1234567890)
```

---

## 6. 实施路线图建议

### Phase 1: 基础改进（1-2 周）
1. 实现 EventStream 类（M1）
2. 添加 AbortController 模式（M3）
3. 使用 Pydantic 替换 dataclass（L2）

### Phase 2: 架构升级（2-3 周）
1. 实现 Provider Registry 模式（S2）
2. 添加消息转换层（M2）
3. 扩展事件系统（S1 的基础部分）

### Phase 3: Agent 能力（3-4 周）
1. 实现 Tool Registry 和 Tool 架构（S3）
2. 添加工具执行三阶段模式（M1）
3. 实现完整的事件生命周期（S1 的完整版）

### Phase 4: 高级特性（可选，2-3 周）
1. 添加 Steering 和 Follow-up 队列（L1）
2. 实现流式 JSON 解析（5.1）
3. 添加跨提供商切换支持

---

## 7. 结论

pi-mono Core Runtime 展示了一个**生产级 Agent 架构**应有的特征：

1. **协议优先设计**：所有提供商统一为标准化事件流
2. **深度类型安全**：TypeBox Schema 实现编译时 + 运行时双重验证
3. **精细生命周期**：10+ 事件类型覆盖完整交互流程
4. **可扩展架构**：注册表模式支持运行时扩展
5. **性能优化**：懒加载、并行执行、信号传播

tui-chatbot 作为一个**轻量级原型**，可以借鉴 pi-mono 的架构思想逐步演进，但考虑到 Python 生态和项目规模，建议：

- **短期**：实现 EventStream（M1）、AbortController（M3）、Pydantic 验证（L2）
- **中期**：引入 Provider Registry（S2）、添加 Tool 架构（S3）
- **长期**：完整的事件系统（S1）、跨提供商支持（M2）

关键借鉴点不在于代码复制，而在于**设计模式的迁移**：流式协议、注册表模式、转换管道、三阶段执行等核心思想都可以在 Python 中优雅实现。
