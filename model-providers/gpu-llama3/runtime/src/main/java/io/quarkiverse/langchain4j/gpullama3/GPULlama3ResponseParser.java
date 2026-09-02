package io.quarkiverse.langchain4j.gpullama3;

import java.util.List;

import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

public class GPULlama3ResponseParser {

    private GPULlama3ResponseParser() {
        // Utility class - prevent instantiation
    }

    public static class ParsedResponse {
        private final String thinkingContent;
        private final String actualResponse;

        /**
         * Creates a new ParsedResponse.
         *
         * @param thinkingContent the thinking content including tags, or null if none
         * @param actualResponse the cleaned response content
         */
        public ParsedResponse(String thinkingContent, String actualResponse) {
            this.thinkingContent = thinkingContent;
            this.actualResponse = actualResponse;
        }

        /**
         * Returns the thinking content including &lt;think&gt; and &lt;/think&gt; tags.
         *
         * @return the thinking content with tags, or null if no thinking content was found
         */
        public String getThinkingContent() {
            return thinkingContent;
        }

        /**
         * Returns the actual response content with thinking tags removed.
         *
         * @return the cleaned response content
         */
        public String getActualResponse() {
            return actualResponse;
        }

        /**
         * Returns true if the response contained thinking content.
         *
         * @return true if thinking content was found, false otherwise
         */
        public boolean hasThinking() {
            return thinkingContent != null && !thinkingContent.trim().isEmpty();
        }
    }

    public static ParsedResponse parseResponse(String rawResponse) {
        if (rawResponse == null) {
            throw new IllegalArgumentException("Raw response cannot be null");
        }

        String thinking = null;
        String actualResponse = rawResponse;

        int thinkStart = rawResponse.indexOf("<think>");
        int thinkEnd = rawResponse.indexOf("</think>");

        // Strip the reasoning block whenever the CLOSING </think> appears, because the opening
        // <think> may be absent from the *generated* text. In thinking-disabled mode the primer
        // puts a <think>…</think> block into the PROMPT, so when a small model still reasons it
        // emits trailing reasoning plus its own </think> without re-opening the tag. The engine
        // never injects a stray </think> (verified: it only ever primes <think> openings), so a
        // lone </think> always marks the end of genuine reasoning. When the opening tag is also
        // present we strip from it; otherwise everything up to and including </think> is reasoning.
        if (thinkEnd != -1) {
            int blockStart = (thinkStart != -1 && thinkStart < thinkEnd) ? thinkStart : 0;
            // Extract thinking content INCLUDING the tags. Preserve the answer's own formatting
            // (newlines, indentation) — never collapse internal whitespace, or code and
            // multi-line answers would be flattened to a single line.
            thinking = rawResponse.substring(blockStart, thinkEnd + 8).trim(); // Include </think>
            String before = rawResponse.substring(0, blockStart);
            String after = rawResponse.substring(thinkEnd + 8); // Skip </think>
            actualResponse = (before + after).trim();
        }

        // Defensively strip any residual control tags the model emitted without a matching pair
        // (e.g. a stray </tool_call> or an extra </think> on small models), so they never leak
        // into the user-facing answer.
        actualResponse = stripControlTags(actualResponse).trim();

        return new ParsedResponse(thinking, actualResponse);
    }

    private static String stripControlTags(String text) {
        return text.replace("<think>", "")
                .replace("</think>", "")
                .replace("<tool_call>", "")
                .replace("</tool_call>", "");
    }

    public static String extractThinking(String rawResponse) {
        return parseResponse(rawResponse).getThinkingContent();
    }

    public static String extractResponse(String rawResponse) {
        return parseResponse(rawResponse).getActualResponse();
    }

    public static StreamingParser createStreamingParser(StreamingChatResponseHandler handler) {
        return new StreamingParser(handler);
    }

    /**
     * Parser for handling streaming responses with real-time thinking content separation.
     * <p>
     * This parser detects thinking content as tokens are generated and routes it to
     * the appropriate handler methods (onPartialThinking vs onPartialResponse).
     * The thinking tags are preserved and streamed as part of the thinking content.
     */
    public static class StreamingParser {
        private static final String TOOL_CALL_OPEN = "<tool_call>";
        private static final String TOOL_CALL_CLOSE = "</tool_call>";
        private static final String PYTHON_TAG = "<|python_tag|>";

        private final StreamingChatResponseHandler handler;
        private final StringBuilder buffer = new StringBuilder();
        private boolean insideThinking = false;
        private boolean insideToolCall = false;
        private boolean insidePythonTagCall = false;
        private final StringBuilder toolCallBuffer = new StringBuilder();
        private final StringBuilder thinkingAccumulator = new StringBuilder();
        private int lastProcessedLength = 0;

        /**
         * Creates a new streaming parser.
         *
         * @param handler the streaming response handler
         */
        public StreamingParser(StreamingChatResponseHandler handler) {
            this.handler = handler;
        }

        /**
         * Processes each event as it is produced by the engine.
         *
         * <p>
         * Two things this used to do are the engine's now, and both were subtly wrong here. Stop
         * tokens no longer need filtering: the façade never emits a terminal control token. And
         * decoding is no longer per token — this used to call
         * {@code tokenizer.decode(List.of(tokenId))} for each id, which returns a replacement
         * character for a token carrying half of a multi-byte character, so a non-ASCII character
         * arrived in the stream as U+FFFD. The engine decodes incrementally and attaches the text to
         * the event that completes it.
         *
         * @param event one emitted completion token: its id, and the text it completed
         */
        public void onEvent(org.beehive.gpullama3.api.GenerationEvent event) {
            if (event.text().isEmpty()) {
                return; // mid-character, or a token with nothing to display
            }
            buffer.append(event.text());
            processNewContent(buffer.toString());
        }

        /**
         * Processes new content in the buffer, detecting thinking and tool-call state
         * transitions and routing content to the appropriate handler methods.
         *
         * <p>
         * Tool-call markers handled:
         * <ul>
         * <li>{@code <tool_call>} / {@code </tool_call>} — LLaMA 3.2 and Qwen3</li>
         * <li>{@code <|python_tag|>} — LLaMA 3.1 (no explicit close tag; resolved in
         * {@link #finish()})</li>
         * </ul>
         * Characters inside a tool-call block are buffered rather than forwarded to the
         * handler, so the client never sees raw tool-call JSON as a partial response.
         */
        private void processNewContent(String currentText) {
            if (currentText.length() <= lastProcessedLength) {
                return; // No new content
            }

            String newContent = currentText.substring(lastProcessedLength);

            // Process each character in the new content
            for (int i = 0; i < newContent.length(); i++) {
                int pos = lastProcessedLength + i;

                // Inside <tool_call>…</tool_call>
                if (insideToolCall) {
                    if (regionMatches(currentText, pos, TOOL_CALL_CLOSE)) {
                        // Buffered rather than parsed. Extraction is the engine's -- the result
                        // carries the calls it validated -- and parsing the same text twice, with
                        // two implementations that can disagree, is how a caller ends up executing
                        // a call the engine did not report.
                        toolCallBuffer.setLength(0);
                        insideToolCall = false;
                        i += TOOL_CALL_CLOSE.length() - 1;
                    } else {
                        toolCallBuffer.append(newContent.charAt(i));
                    }
                    continue;
                }

                // Inside <|python_tag|>… (no close tag, resolved in finish())
                if (insidePythonTagCall) {
                    toolCallBuffer.append(newContent.charAt(i));
                    continue;
                }

                // Thinking open
                if (!insideThinking && isStartOfThinkTag(currentText, pos)) {
                    insideThinking = true;
                    // Stream the opening tag as thinking
                    thinkingAccumulator.append("<think>");
                    handler.onPartialThinking(new PartialThinking("<think>"));
                    i += 6; // Skip the rest of "<think>"
                    continue;
                }

                // Thinking close
                if (insideThinking && isStartOfEndThinkTag(currentText, pos)) {
                    // Stream the closing tag as thinking
                    thinkingAccumulator.append("</think>");
                    handler.onPartialThinking(new PartialThinking("</think>"));
                    insideThinking = false;
                    i += 7; // Skip the rest of "</think>"
                    continue;
                }

                // Tool call open (<tool_call>)
                if (!insideThinking && regionMatches(currentText, pos, TOOL_CALL_OPEN)) {
                    insideToolCall = true;
                    i += TOOL_CALL_OPEN.length() - 1;
                    continue;
                }

                // LLaMA 3.1 python tag (<|python_tag|>)
                if (!insideThinking && regionMatches(currentText, pos, PYTHON_TAG)) {
                    insidePythonTagCall = true;
                    i += PYTHON_TAG.length() - 1;
                    continue;
                }

                // Route the character to appropriate handler
                char c = newContent.charAt(i);
                if (insideThinking) {
                    thinkingAccumulator.append(c);
                    handler.onPartialThinking(new PartialThinking(String.valueOf(c)));
                } else {
                    handler.onPartialResponse(String.valueOf(c));
                }
            }

            lastProcessedLength = currentText.length();
        }

        /**
         * Must be called after the model finishes generating tokens.
         *
         * <p>
         * Resolves an unclosed {@code <|python_tag|>} block (LLaMA 3.1), which has no close tag
         * and is only complete when generation stops.
         *
         * <p>
         * It no longer returns the calls it saw. The engine's result carries the calls it
         * validated, and this parser's job is to keep tool JSON out of the <i>visible</i> stream —
         * not to extract it a second time with an implementation that could disagree.
         *
         * @return whether a tool-call block was still open when generation ended
         */
        public boolean finish() {
            return insidePythonTagCall && !toolCallBuffer.isEmpty();
        }

        /**
         * Returns the thinking content accumulated during generation (including
         * {@code <think>} and {@code </think>} tags), or {@code null} if no
         * thinking block was emitted.
         */
        public String getThinkingContent() {
            String s = thinkingAccumulator.toString().strip();
            return s.isEmpty() ? null : s;
        }

        private static boolean regionMatches(String text, int start, String marker) {
            return start + marker.length() <= text.length()
                    && text.regionMatches(start, marker, 0, marker.length());
        }

        /**
         * Checks if the text at the given position starts with "&lt;think&gt;".
         */
        private boolean isStartOfThinkTag(String text, int position) {
            return position + 7 <= text.length() && text.regionMatches(position, "<think>", 0, 7);
        }

        /**
         * Checks if the text at the given position starts with "&lt;/think&gt;".
         */
        private boolean isStartOfEndThinkTag(String text, int position) {
            return position + 8 <= text.length() && text.regionMatches(position, "</think>", 0, 8);
        }
    }
}
