package io.quarkiverse.langchain4j.gpullama3;

import java.util.List;
import java.util.function.Consumer;

import org.beehive.gpullama3.api.GenerationEvent;
import org.beehive.gpullama3.api.GenerationRequest;
import org.beehive.gpullama3.api.GenerationResult;
import org.beehive.gpullama3.api.LocalModel;
import org.jboss.logging.Logger;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.ChatRequest;

/**
 * Shared generation for the chat and streaming beans.
 *
 * <p>
 * Migrated to the engine's public façade (GPULlama3 T12.9b). It used to assemble the Llama 3.2
 * chat template by hand — beginning-of-text, tool definitions into the first user message, an
 * environment prefix on the system turn, the thinking-control primer, tool-aware stop tokens — all
 * of which is the engine's now. This class translates vocabulary and nothing else.
 *
 * <p>
 * It does not own the model or the session: {@link GPULlama3ModelHolder} does, and closes them
 * once at application shutdown.
 */
abstract class GPULlama3BaseModel {

    private static final Logger LOG = Logger.getLogger(GPULlama3BaseModel.class);

    /**
     * Centralized holder of the loaded model and the session both beans generate through.
     * *Shared* across ChatModel and StreamingChatModel instances. Lazily initialized by
     * ensureInitialized() when first used.
     */
    GPULlama3ModelHolder holder;

    public LocalModel getModel() {
        return holder.model;
    }

    /**
     * Runs inference for the given {@link ChatRequest}.
     *
     * @param onEvent receives one ordered event per emitted completion token — its id and the text
     *        it completed — or {@code null} for a non-streaming call
     */
    public GenerationResult modelResponse(ChatRequest request, Consumer<GenerationEvent> onEvent) {
        holder.ensureInitialized();

        List<ToolSpecification> tools = request.toolSpecifications();

        GenerationRequest.Builder builder = GenerationRequest.builder()
                .messages(GPULlama3Conversions.toEngineMessages(request.messages()))
                .maxNewTokens(holder.maxTokens);
        if (tools != null && !tools.isEmpty()) {
            builder.tools(GPULlama3Conversions.toEngineTools(tools));
            if (LOG.isInfoEnabled()) {
                LOG.infof("[Tool turn] %d tool(s) available: %s", tools.size(),
                        tools.stream().map(ToolSpecification::name).toList());
            }
        }
        if (onEvent != null) {
            builder.onEvent(onEvent);
        }

        GenerationResult result = holder.session.generate(builder.build());
        LOG.debugf("finishReason=%s generatedTokens=%d toolCalls=%d raw response: >>>%s<<<",
                result.finishReason(), result.generatedTokens(), result.toolCalls().size(),
                result.text());
        return result;
    }

    protected static String generateCallId() {
        return GPULlama3Conversions.generateCallId();
    }

    protected static String normalizeJson(String json) {
        return GPULlama3Conversions.normalizeJson(json);
    }
}
