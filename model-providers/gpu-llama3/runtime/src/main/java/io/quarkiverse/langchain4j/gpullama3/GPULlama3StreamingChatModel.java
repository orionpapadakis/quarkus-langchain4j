package io.quarkiverse.langchain4j.gpullama3;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static io.quarkiverse.langchain4j.runtime.VertxUtil.runOutEventLoop;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

import org.beehive.gpullama3.model.format.ToolCallExtract;
import org.jboss.logging.Logger;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.internal.ChatRequestValidationUtils;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;

/**
 * GPULlama3StreamingChatModel is a specialized implementation of the {@link StreamingChatModel} for Quarkus-Langchain4j
 * extension.
 * It enables streaming mode for GPULlama3.java integration.
 * <p>
 * Considering that interoperability with GPU sometimes can be latency-prone, it operates in
 * an asynchronous, non-blocking manner, enabling efficient handling of conversational requests.
 * </p>
 *
 * <p>
 * Initialization:
 * </p>
 * <ul>
 * <li>The initialization of the model is performed in a background thread and marked complete using a future</li>
 * <li>If an inference request is made prior to initialization, it waits for the process to complete</li>
 * </ul>
 *
 * <p>
 * Response Generation:
 * </p>
 * <ul>
 * <li>Processes user inputs and generates model response</li>
 * <li>It is non-blocking as it is driven by a background thread</li>
 * <li>Delivers responses in a streaming format through registered handlers</li>
 * </ul>
 */
public class GPULlama3StreamingChatModel extends GPULlama3BaseModel implements StreamingChatModel {

    private static final Logger LOG = Logger.getLogger(GPULlama3StreamingChatModel.class);

    private final Builder builderConfig;
    private volatile boolean initialized = false;

    private GPULlama3StreamingChatModel(Builder builder, boolean lazy) {
        if (lazy) {
            this.builderConfig = builder;
            // Don't initialize yet!
        } else {
            this.builderConfig = null;
            // Original background initialization
            runOutEventLoop(() -> {
                LOG.debug("Starting GPULlama3 StreamingChatModel initialization on worker thread");
                doInitialization(builder);
                initialized = true;
            });
        }
    }

    private GPULlama3StreamingChatModel(Builder builder) {
        this(builder, false); // Default to original background initialization
    }

    // Add factory method for lazy initialization
    public static GPULlama3StreamingChatModel createLazy(Builder builder) {
        return new GPULlama3StreamingChatModel(builder, true);
    }

    private void ensureInitialized() {
        if (!initialized && builderConfig != null) {
            synchronized (this) {
                if (!initialized) {
                    LOG.debug("Lazy initialization of GPULlama3StreamingChatModel");
                    doInitialization(builderConfig);
                    initialized = true;
                }
            }
        }
    }

    private void doInitialization(Builder builder) {
        GPULlama3ModelRegistry gpuLlama3ModelRegistry = GPULlama3ModelRegistry.getOrCreate(builder.modelCachePath);
        try {
            Path modelPath = gpuLlama3ModelRegistry.downloadModel(builder.modelName, builder.quantization,
                    Optional.empty(), Optional.empty());
            defineDefaultConfigForModel(builder.modelName);
            Double temp = getOrDefault(builder.temperature, defaultTemperature);
            Double topP = getOrDefault(builder.topP, defaultTopP);
            Integer seed = getOrDefault(builder.seed, ThreadLocalRandom.current().nextInt());
            Integer maxTokens = getOrDefault(builder.maxTokens, defaultMaxTokens);
            Boolean onGPU = getOrDefault(builder.onGPU, Boolean.TRUE);

            LOG.debugf("GPULlama3StreamingChatModel init: modelPath=%s temperature=%s topP=%s seed=%s maxTokens=%s onGPU=%s",
                    modelPath, temp, topP, seed, maxTokens, onGPU);

            init(modelPath, temp, topP, seed, maxTokens, onGPU);
            LOG.infof("Streaming model loaded: %s (onGPU=%s)", modelPath.getFileName(), onGPU);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
        ChatRequestValidationUtils.validateMessages(chatRequest.messages());
        ChatRequestParameters parameters = chatRequest.parameters();
        ChatRequestValidationUtils.validateParameters(parameters);
        ChatRequestValidationUtils.validate(parameters.toolChoice());
        ChatRequestValidationUtils.validate(parameters.responseFormat());

        // Run the GPU operations on a worker thread using runOutEventLoop
        runOutEventLoop(() -> {
            try {
                ensureInitialized(); // Build happens HERE on first call!
                LOG.debug("Executing GPU Llama inference on worker thread");
                coreDoChat(chatRequest, handler);
            } catch (Exception e) {
                LOG.error("Failed during lazy initialization or inference", e);
                handler.onError(e);
            }
        });
    }

    /**
     * The actual doChat logic.
     * It is called by a worker thread in a non-blocking manner.
     */
    private void coreDoChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
        try {
            // The StreamingParser detects and buffers tool-call JSON in real time (<tool_call> and
            // <|python_tag|> markers), so streaming is never suppressed — plain-text responses
            // stream token-by-token even when tool specifications are registered.
            GPULlama3ResponseParser.StreamingParser parser = GPULlama3ResponseParser.createStreamingParser(handler, getModel());

            String rawResponse = modelResponse(chatRequest, parser::onToken);

            // Finalize parser: resolves any unclosed <|python_tag|> tool call (LLaMA 3.1)
            List<ToolCallExtract> toolCalls = parser.finish();

            // Check for tool calls
            // Fallback for models that emit raw JSON without <tool_call> tags (rare)
            if (toolCalls.isEmpty()) {
                toolCalls = chatFormat.extractAllToolCalls(rawResponse);
            }

            if (!toolCalls.isEmpty()) {
                LOG.infof("[LLM → tool call]\n%s", rawResponse.strip());
                String thinkingContent = parser.getThinkingContent();
                List<ToolExecutionRequest> toolReqs = new ArrayList<>();
                for (ToolCallExtract tc : toolCalls) {
                    String callId = tc.id().orElseGet(() -> generateCallId());
                    LOG.infof("[Tool call] → %s(%s)", tc.name(),
                            tc.argumentsJson().replace("\n", "").replaceAll("\\s+", " "));
                    toolReqs.add(ToolExecutionRequest.builder()
                            .id(callId)
                            .name(tc.name())
                            .arguments(tc.argumentsJson())
                            .build());
                }
                handler.onCompleteResponse(ChatResponse.builder()
                        .aiMessage(AiMessage.builder()
                                .thinking(thinkingContent)
                                .toolExecutionRequests(toolReqs)
                                .build())
                        .finishReason(FinishReason.TOOL_EXECUTION)
                        .build());
                return;
            }

            // Plain text — parse thinking and deliver final response
            GPULlama3ResponseParser.ParsedResponse parsed = GPULlama3ResponseParser.parseResponse(rawResponse);

            LOG.infof("[LLM response]\n%s", parsed.getActualResponse());

            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(AiMessage.builder()
                            .text(parsed.getActualResponse())
                            .thinking(parsed.getThinkingContent())
                            .build())
                    .build());
        } catch (Exception e) {
            LOG.error("Error in GPULlama3 coreDoChat", e);
            handler.onError(e);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private Optional<Path> modelCachePath;
        private String modelName = Consts.DEFAULT_CHAT_MODEL_NAME;
        private String quantization = Consts.DEFAULT_CHAT_MODEL_QUANTIZATION;
        protected Double temperature;
        protected Double topP;
        protected Integer seed;
        protected Integer maxTokens;
        protected Boolean onGPU;

        public Builder() {
            // This is public so it can be extended
        }

        public Builder modelCachePath(Optional<Path> modelCachePath) {
            this.modelCachePath = modelCachePath;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder onGPU(Boolean onGPU) {
            this.onGPU = onGPU;
            return this;
        }

        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }

        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder seed(Integer seed) {
            this.seed = seed;
            return this;
        }

        public GPULlama3StreamingChatModel build() {
            return new GPULlama3StreamingChatModel(this);
        }
    }
}
