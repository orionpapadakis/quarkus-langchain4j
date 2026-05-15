package io.quarkiverse.langchain4j.gpullama3;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.IntConsumer;
import java.util.stream.Collectors;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ToolCallExtract;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.jboss.logging.Logger;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.internal.Json;
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.model.chat.request.ChatRequest;

abstract class GPULlama3BaseModel {

    private static final Logger LOG = Logger.getLogger(GPULlama3BaseModel.class);

    List<Integer> promptTokens;
    ChatFormat chatFormat;
    private Integer maxTokens;
    private Boolean onGPU;
    private Model model;
    private Sampler sampler;
    private State state;
    private TornadoVMMasterPlan tornadoVMPlan;

    // @formatter:off
    public void init(
            Path modelPath,
            Double temperature,
            Double topP,
            Integer seed,
            Integer maxTokens,
            Boolean onGPU) {
        this.maxTokens = maxTokens;
        this.onGPU = onGPU;

        try {
            this.model = ModelLoader.loadModel(modelPath, maxTokens, true, onGPU);
            this.state = model.createNewState();
            this.sampler = Sampler.selectSampler(
                    model.configuration().vocabularySize(), temperature.floatValue(), topP.floatValue(), seed);
            this.chatFormat = model.chatFormat();
            if (onGPU) {
                this.tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from " + modelPath, e);
        }
    }

    public Model getModel() {
        return model;
    }

    public Sampler getSampler() {
        return sampler;
    }

    /**
     * Runs inference for the given {@link ChatRequest} and returns the raw decoded response text.
     *
     * When the request contains tool specifications, the tool definitions are injected into the
     * system message and tool-aware stop tokens are used so the model can signal a tool call.
     * The caller ({@link GPULlama3ChatModel} / {@link GPULlama3StreamingChatModel}) is responsible
     * for inspecting the raw text and deciding whether to return a tool execution request or a
     * plain text response.
     */
    public String modelResponse(ChatRequest request, IntConsumer tokenConsumer) {
        this.promptTokens = new ArrayList<>();

        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }

        // Build tools JSON if the request carries tool definitions
        List<ToolSpecification> tools = request.toolSpecifications();
        String toolsJson = (tools != null && !tools.isEmpty()) ? buildToolsJson(tools) : null;

        if (toolsJson != null && !chatFormat.supportsToolCalling()) {
            throw new UnsupportedOperationException(
                    "Tool calling is not supported for model format: " + chatFormat.getClass().getSimpleName());
        }

        processPromptMessages(request.messages(), toolsJson);

        if (toolsJson != null) {
            boolean hasPriorToolResult = request.messages().stream()
                    .anyMatch(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.TOOL_EXECUTION_RESULT);
            String toolNames = tools.stream()
                    .map(ToolSpecification::name)
                    .collect(Collectors.joining(", "));
            long priorResultCount = request.messages().stream()
                    .filter(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.TOOL_EXECUTION_RESULT)
                    .count();
            if (hasPriorToolResult) {
                LOG.infof("[Tool turn] %d tool(s) available: %s  (after %d result(s))",
                        tools.size(), toolNames, priorResultCount);
            } else {
                LOG.infof("[Tool turn] %d tool(s) available: %s", tools.size(), toolNames);
            }
        }

        // Use tool-aware stop tokens whenever tools are present so the model can signal a tool
        // call (eom_id on LLaMA 3.1) or a regular response (eot_id) on every turn — including
        // turns that already contain prior tool results.
        Set<Integer> stopTokens = (toolsJson != null)
                ? chatFormat.getToolAwareStopTokens()
                : chatFormat.getStopTokens();

        List<Integer> responseTokens;

        if (onGPU) {
            responseTokens = model.generateTokensGPU(
                    state,
                    0,
                    promptTokens.subList(0, promptTokens.size()),
                    stopTokens,
                    maxTokens,
                    sampler,
                    false,
                    tokenConsumer,
                    tornadoVMPlan);
        } else {
            responseTokens = model.generateTokens(
                    state,
                    0,
                    promptTokens.subList(0, promptTokens.size()),
                    stopTokens,
                    maxTokens,
                    sampler,
                    false,
                    tokenConsumer);
        }

        Integer stopToken = null;
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            stopToken = responseTokens.getLast();
            responseTokens.removeLast();
        }

        String responseText = model.tokenizer().decode(responseTokens);

        // Append to conversation history
        promptTokens.addAll(responseTokens);

        // Add the stop token to complete the message
        if (stopToken != null) {
            promptTokens.add(stopToken);
        }

        LOG.debug("stopToken=" + stopToken + "  responseTokens=" + responseTokens.size()
                + "  raw response: >>>" + responseText + "<<<");

        if (stopToken == null) {
            return "Ran out of context length...\n Increase context length with by passing to llama-tornado --max-tokens XXX";
        } else {
            return responseText;
        }
    }
    // @formatter:on

    /**
     * Encodes all conversation messages into {@code promptTokens} using the <b>native Llama 3.2
     * chat template</b> extracted from the GGUF metadata.
     *
     * <p>
     * Key template behaviours reproduced here:
     * <ul>
     * <li>System message gets an {@code "Environment: ipython\n"} prefix when tools are active.</li>
     * <li>Tool definitions are injected into the <em>first</em> user message
     * ({@code tools_in_user_message = true} is the default in the template).</li>
     * <li>Assistant tool-call turns are encoded as native JSON:
     * {@code {"name":"…","parameters":{…}}} — not {@code <tool_call>} XML.</li>
     * <li>Tool results use the {@code ipython} role (handled by
     * {@link ChatFormat#encodeToolResultTurn}).</li>
     * </ul>
     */
    private void processPromptMessages(List<ChatMessage> messageList, String toolsJson) {
        boolean toolsInjected = false;

        // Tool definitions are injected on every turn so the model always knows which tools
        // are available — matching Ollama's behaviour of sending the tools array with every
        // request. Models correctly decide when to call a tool vs. synthesise a final answer
        // based on whether they already have enough information from prior tool results.
        boolean hasPriorToolResult = messageList.stream()
                .anyMatch(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.TOOL_EXECUTION_RESULT);
        boolean injectTools = toolsJson != null;
        boolean userMessageInjection = injectTools && chatFormat.injectsToolsInUserMessage();

        LOG.debug("processPromptMessages: msgs=" + messageList.size()
                + "  injectTools=" + injectTools + "  userMsgInjection=" + userMessageInjection);

        for (ChatMessage msg : messageList) {
            switch (msg.type()) {
                case SYSTEM -> {
                    SystemMessage systemMessage = (SystemMessage) msg;
                    if (model.shouldAddSystemPrompt()) {
                        String content = systemMessage.text();
                        if (injectTools) {
                            if (userMessageInjection) {
                                String prefix = chatFormat.toolSystemMessagePrefix();
                                if (!prefix.isEmpty())
                                    content = prefix + content;
                            } else {
                                content = content + chatFormat.toolSystemPromptSuffix(toolsJson);
                                toolsInjected = true;
                            }
                        }
                        LOG.debugf("SYSTEM (first 300): %s",
                                content.substring(0, Math.min(300, content.length())));
                        promptTokens.addAll(
                                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, content)));
                    }
                }
                case USER -> {
                    UserMessage userMessage = (UserMessage) msg;
                    String userText = userMessage.singleText();
                    if (injectTools && !toolsInjected && userMessageInjection) {
                        userText = chatFormat.toolFirstUserMessagePrefix(toolsJson) + userText;
                        toolsInjected = true;
                    }
                    LOG.debugf("USER (first 200): %s", userText.substring(0, Math.min(200, userText.length())));
                    promptTokens.addAll(chatFormat.encodeMessage(
                            new ChatFormat.Message(ChatFormat.Role.USER, userText)));
                }
                case AI -> {
                    AiMessage aiMessage = (AiMessage) msg;
                    if (aiMessage.hasToolExecutionRequests()) {
                        List<ToolCallExtract> toolCalls = aiMessage.toolExecutionRequests().stream()
                                .map(req -> new ToolCallExtract(req.name(), req.arguments()))
                                .collect(Collectors.toList());
                        promptTokens.addAll(chatFormat.encodeToolCallAssistantTurn(toolCalls));
                    } else if (aiMessage.text() != null) {
                        promptTokens.addAll(chatFormat.encodeMessage(
                                new ChatFormat.Message(ChatFormat.Role.ASSISTANT, aiMessage.text())));
                    }
                }
                case TOOL_EXECUTION_RESULT -> {
                    ToolExecutionResultMessage toolMessage = (ToolExecutionResultMessage) msg;
                    String resultText = unwrapToolResult(toolMessage.text());
                    LOG.infof("[Tool result] ← %s:\n%s", toolMessage.toolName(), resultText);
                    promptTokens.addAll(chatFormat.encodeToolResultTurn(
                            toolMessage.id(), toolMessage.toolName(), resultText));
                }
                default -> {
                    // Unsupported message types are silently skipped
                }
            }
        }

        // Fallback: no system or user message encountered — inject tools at the start
        if (injectTools && !toolsInjected && !userMessageInjection && model.shouldAddSystemPrompt()) {
            String toolsOnlySystem = chatFormat.toolSystemPromptSuffix(toolsJson).stripLeading();
            promptTokens.addAll(0,
                    chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, toolsOnlySystem)));
        }

        // Prime the model to start generating an assistant response
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
    }

    private static final String CALL_ID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789";

    /** Generates an Ollama-style tool call ID: {@code call_} + 8 random alphanumeric chars. */
    protected static String generateCallId() {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        StringBuilder sb = new StringBuilder("call_");
        for (int i = 0; i < 8; i++) {
            sb.append(CALL_ID_CHARS.charAt(rng.nextInt(CALL_ID_CHARS.length())));
        }
        return sb.toString();
    }

    /**
     * Builds the tools JSON array in OpenAI / native Llama 3.2 format:
     * {@code [{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]}.
     *
     * This matches the format Ollama sends to the model and the format Llama 3.2 expects in its
     * native {@code <|start_header_id|>tools<|end_header_id|>} section. Using this format (rather
     * than flat JSON without the {@code type/function} wrapper) triggers reliable tool calling.
     */
    static String buildToolsJson(List<ToolSpecification> tools) {
        List<Map<String, Object>> toolArray = new ArrayList<>();
        for (ToolSpecification tool : tools) {
            Map<String, Object> funcMap = new LinkedHashMap<>();
            funcMap.put("name", tool.name());
            if (tool.description() != null) {
                funcMap.put("description", tool.description());
            }
            funcMap.put("parameters", buildParametersMap(tool));

            Map<String, Object> toolMap = new LinkedHashMap<>();
            toolMap.put("type", "function");
            toolMap.put("function", funcMap);
            toolArray.add(toolMap);
        }
        return Json.toJson(toolArray);
    }

    private static Map<String, Object> buildParametersMap(ToolSpecification tool) {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        Map<String, Object> props = new LinkedHashMap<>();
        if (tool.parameters() != null) {
            for (var entry : tool.parameters().properties().entrySet()) {
                props.put(entry.getKey(), JsonSchemaElementUtils.toMap(entry.getValue()));
            }
            List<String> required = tool.parameters().required();
            if (required != null && !required.isEmpty()) {
                params.put("required", required);
            }
        }
        params.put("properties", props);
        return params;
    }

    /**
     * LangChain4j serializes {@code String} tool results via {@code Json.toJson()}, which wraps
     * them in a JSON string literal: {@code "hello\nworld"} becomes {@code "\"hello\\nworld\""}.
     * Unwrap the JSON quoting so the model receives plain readable text with real newlines.
     */
    private static String unwrapToolResult(String text) {
        if (text == null)
            return "";
        if (text.startsWith("\"")) {
            try {
                return Json.fromJson(text, String.class);
            } catch (Exception ignored) {
            }
        }
        return text;
    }
}
