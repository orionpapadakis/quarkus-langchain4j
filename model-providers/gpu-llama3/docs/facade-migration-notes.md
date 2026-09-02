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

## Known: the default `prefill-decode=true` configuration does not work, on either path

With the extension's published default (`prefill-decode=true`, `prefill-batch-size=32`) the sample
app returns HTTP 500 **before and after** this migration, on this machine and fixture:

| Configuration | Legacy extension | Migrated extension |
| --- | --- | --- |
| `prefill-decode=true` (default) | `TornadoOutOfMemoryException: Unable to allocate 525336592 bytes` | `TornadoRuntimeException: null object passed into streamIn() in schedule prefillActivation` |
| `prefill-decode=false` | works | works (blocking **and** streaming) |

**Both fail; they fail differently, and the difference is explained.** The legacy path set
`llama.prefillBatchSize` as a JVM-global property before the model loaded, which sized the state's
batch arrays; the façade takes an `ExecutionPolicy` on `ModelOptions`, and the session's state is
created with no batch size — so the batched plan is built against a state that has no batch arrays
and fails earlier, at plan construction, instead of later, at device allocation.

That is an **engine gap**, recorded rather than worked around: batched prefill is
[DEF-01](https://github.com/beehive-lab/GPULlama3.java) territory — unverified, not enabled by
default, and its bounds are not to be widened to make a test pass.
