# Façade migration notes (GPULlama3 T12.9b)

## What changed

The extension used the engine's internals — `ModelLoader`, `State`, `Sampler`, `ChatFormat`,
`ToolCallExtract`, `ToolCallParserUtils`, `TornadoVMMasterPlan` — and assembled the Llama 3.2 chat
template itself. It now uses the engine's public façade only:

| Was | Is |
| --- | --- |
| `ModelLoader.loadModel` + a hand-built `TornadoVMMasterPlan` | `LocalModels.load` |
| a shared `State` and hand-managed `promptTokens` history | one `GenerationSession` |
| `ChatFormat` encoding, per-family tool JSON, tool-aware stop tokens | `ChatMessage` / `ToolSpec` |
| `IntConsumer` of raw token ids | `GenerationEvent` (id **and** the text it completed) |
| `chatFormat.extractAllToolCalls(...)` | `GenerationResult.toolCalls()` |
| `chatFormat.encodeThinkingControl(...)` | `ThinkingMode` on the session |

## Ownership

`GPULlama3ModelHolder` is the single owner. It loads the model, opens the one session both beans
generate through, and closes both **exactly once** at application shutdown, via a new
`ShutdownContext` task. Requests borrow the session and do not own it.

**Before this, nothing closed anything**: the TornadoVM plan and its device copy of the weights lived
until the JVM exited.

One session, not one per request: on the accelerator path each session builds its own plan holding
its own device copy of the weights, so a session per request exhausts device memory.

## Behaviour preserved, and behaviour changed

**`enable-thinking` is preserved.** It maps to `ThinkingMode.ENABLED`/`DISABLED`. The façade rejects
an explicit mode on a family with no reasoning phase — deliberately, so a caller who turned thinking
off cannot silently get it anyway. This extension's property has always been a *no-op* on such
families rather than an error, so the mode is attempted once at initialization and a family that
cannot represent it falls back to the default with the same message this class already logged.
Verified in the running app on Llama-3.2-1B: *"Thinking control not applicable for this model;
enable-thinking=false has no effect"*.

**Tool prompt rendering moved to the engine, and this is a change.** The extension used to build
*family-specific* tool JSON (`buildToolsJsonLlama` / `buildToolsJsonQwen3`, switching on
`ModelType`). The engine's chat formats now render tool definitions, which is where that decision
belongs — an extension second-guessing the format is how the two drift. The tool *text* a model sees
therefore differs from before.

**The streaming parser no longer extracts tool calls.** It still keeps tool JSON out of the visible
stream; the calls come from `GenerationResult.toolCalls()`, which the engine validated. Parsing the
same text twice with two implementations that can disagree is how a caller ends up executing a call
the engine did not report.

**Incremental decoding is a fix.** `StreamingParser.onToken` decoded one id at a time
(`tokenizer.decode(List.of(id))`), which returns U+FFFD for a token carrying half a multi-byte
character. The engine decodes incrementally and attaches the text to the event that completes it.

## Verified

- **`GPULlama3ConversionsTest` — 15 cases, no model, no device, no container.** Calls with and
  without arguments, ordered calls with distinct ids, results mapped back with matching ids, a full
  round trip, tool specifications, and the stop-reason mapping (`TOOL_CALL` → `TOOL_EXECUTION`, a
  truncated call staying `LENGTH`).
- **The sample app, end to end, on the accelerator.** `/chat/blocking` returns a correct answer and
  `/chat/streaming` streams token by token.

The conversions use plain Jackson rather than `dev.langchain4j.internal.Json`, which resolves to
Quarkus's codec factory and needs the CDI container — a conversion that cannot be tested without
booting Quarkus is not a pure function.

## Batched prefill: default-off, and the engine gap that made it fail

The extension used to default `prefill-decode=true` with `prefill-batch-size=32`, so every
deployment ran batched prefill without asking for it. It is now `prefill-decode=false` and
`prefill-batch-size=1`. Batched prefill is default-off in the engine on every backend, and
turning it on is an opt-in that needs its own performance evidence for the model and device.

Enabling it used to fail with `TornadoRuntimeException: null object passed into streamIn() in
schedule prefillActivation`. The batch arrays are sized when the state is allocated, from
`llama.prefillBatchSize`; the facade supplies the width as an `ExecutionPolicy` on
`ModelOptions`, which is resolved after allocation. So the batched plan was built against a
state that had no batch arrays. The engine now scopes the width around state construction,
and gates it with `BatchedPrefillPolicyAccelTest`.

Verified after both changes, on CUDA with TornadoVM 6.0.0 and JDK 25:
`chat-demo` and `streaming-demo` run to completion with the defaults, and `tool-demo-ls`
runs with `prefill-decode=true` and `prefill-batch-size=32` explicitly set, emitting a tool
call, executing it and returning.

## The backend is the SDK's to choose

`GPULlama3ModelHolder` named `BackendId.CUDA` whenever `onGPU` was true. The engine rejects an
explicit backend that disagrees with the device it resolves — deliberately, rather than
silently running on a different one — so the extension could not start on an OpenCL or Metal
SDK. It now sets `use.tornadovm` and names no backend, leaving the choice to the installed SDK.
