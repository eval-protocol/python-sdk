import type { Message } from "../types/eval-protocol";
import { useState } from "react";
import Button from "./Button";
import { Tooltip } from "./Tooltip";

export const MessageBubble = ({ message }: { message: Message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [toolCallCopySuccess, setToolCallCopySuccess] = useState<
    Record<number, boolean>
  >({});
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isTool = message.role === "tool";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;
  const hideMessageContent = message.role === "assistant" && hasToolCalls;

  // Get the message content as a string
  const reasoning = (message as any).reasoning_content as string | undefined;
  const titleLabel =
    message.role === "system" && message.name ? message.name : message.role;
  const getMessageContent = () => {
    if (typeof message.content === "string") {
      return message.content;
    } else if (Array.isArray(message.content)) {
      return message.content
        .map((part) => {
          if (part.type === "text") return part.text;
          if (part.type === "image_url") return "[Image]";
          return JSON.stringify(part);
        })
        .join("");
    } else {
      return JSON.stringify(message.content);
    }
  };

  const messageContent = hideMessageContent ? "" : getMessageContent();
  const hasMessageContent = messageContent.trim().length > 0;
  const isLongMessage = messageContent.length > 200; // Threshold for considering a message "long"

  const renderContent = () => {
    if (hideMessageContent) {
      return null;
    }
    if (typeof message.content === "string") {
      return isLongMessage && !isExpanded
        ? message.content.substring(0, 200) + "..."
        : message.content;
    } else if (Array.isArray(message.content)) {
      let currentLength = 0;
      const parts = [];
      const limit = 200;

      for (let i = 0; i < message.content.length; i++) {
        const part = message.content[i];
        
        if (!isExpanded && currentLength >= limit) {
          break;
        }

        if (part.type === "image_url") {
          parts.push(
            <div key={i} className="my-2">
              <img 
                src={part.image_url.url} 
                alt="Trace content" 
                className="max-w-full h-auto rounded border border-gray-200"
                style={{ maxHeight: '300px' }}
              />
            </div>
          );
        } else if (part.type === "text") {
          const text = part.text;
          if (!isExpanded && currentLength + text.length > limit) {
            const remaining = limit - currentLength;
            if (remaining > 0) {
              parts.push(<span key={i}>{text.substring(0, remaining)}...</span>);
            }
            currentLength += text.length;
            break; 
          } else {
            parts.push(<span key={i}>{text}</span>);
            currentLength += text.length;
          }
        } else {
          const str = JSON.stringify(part);
          parts.push(<span key={i}>{str}</span>);
          currentLength += str.length;
        }
      }
      return <div className="flex flex-col">{parts}</div>;
    } else {
      return JSON.stringify(message.content);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(messageContent);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error("Failed to copy message:", err);
    }
  };

  const handleToolCallCopy = async (index: number, argumentsText: string) => {
    try {
      await navigator.clipboard.writeText(argumentsText);
      setToolCallCopySuccess((prev) => ({ ...prev, [index]: true }));
      setTimeout(() => {
        setToolCallCopySuccess((prev) => {
          const newState = { ...prev };
          delete newState[index];
          return newState;
        });
      }, 2000);
    } catch (err) {
      console.error("Failed to copy tool call:", err);
    }
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-1`}>
      <div
        className={`max-w-sm lg:max-w-md xl:max-w-lg px-2 py-1 border text-xs relative ${
          isUser
            ? "bg-blue-50 border-blue-200 text-blue-900"
            : isSystem
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : isTool
            ? "bg-green-50 border-green-200 text-green-900"
            : "bg-yellow-50 border-yellow-200 text-yellow-900"
        }`}
      >
        {/* Copy button positioned in top-right corner */}
        {hasMessageContent && (
          <div className="absolute top-1 right-1">
            <Tooltip
              content={copySuccess ? "Copied!" : "Copy message to clipboard"}
              position="top"
            >
              <Button
                onClick={handleCopy}
                size="sm"
                variant="secondary"
                className={`p-0.5 h-5 text-[10px] opacity-60 hover:opacity-100 transition-opacity cursor-pointer ${
                  isUser
                    ? "text-blue-600 hover:bg-blue-100"
                    : isSystem
                    ? "text-gray-600 hover:bg-gray-100"
                    : isTool
                    ? "text-green-600 hover:bg-green-100"
                    : "text-yellow-600 hover:bg-yellow-100"
                }`}
              >
                Copy
              </Button>
            </Tooltip>
          </div>
        )}

        <div
          className={`font-semibold text-xs mb-0.5 capitalize ${
            hasMessageContent ? "pr-8" : ""
          }`}
        >
          {titleLabel}
        </div>
        <div className="whitespace-pre-wrap break-words overflow-hidden text-xs">
          {renderContent()}
        </div>
        {isLongMessage && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`mt-1 text-xs underline hover:no-underline ${
              isUser
                ? "text-blue-700"
                : isSystem
                ? "text-gray-600"
                : isTool
                ? "text-green-700"
                : "text-yellow-700"
            }`}
          >
            {isExpanded ? "Show less" : "Show more"}
          </button>
        )}
        {reasoning && reasoning.trim().length > 0 && (
          <div
            className={`mt-2 pt-1 border-t ${
              isTool ? "border-green-200" : "border-yellow-200"
            }`}
          >
            <div
              className={`font-semibold text-xs mb-0.5 ${
                isTool ? "text-green-700" : "text-yellow-700"
              }`}
            >
              Thinking:
            </div>
            <details className="mb-1">
              <summary
                className={`cursor-pointer text-xs ${
                  isTool ? "text-green-700" : "text-yellow-700"
                }`}
              >
                Show reasoning
              </summary>
              <pre
                className={`mt-1 p-1 border rounded text-xs whitespace-pre-wrap break-words ${
                  isTool
                    ? "bg-green-100 border-green-200 text-green-800"
                    : "bg-yellow-100 border-yellow-200 text-yellow-800"
                }`}
              >
                {reasoning}
              </pre>
            </details>
          </div>
        )}
        {hasToolCalls && message.tool_calls && (
          <div
            className={`mt-2 pt-1 border-t ${
              isTool ? "border-green-200" : "border-yellow-200"
            }`}
          >
            <div
              className={`font-semibold text-xs mb-0.5 ${
                isTool ? "text-green-700" : "text-yellow-700"
              }`}
            >
              Tool Calls:
            </div>
            {message.tool_calls.map((call, i) => {
              const hasToolCallArguments =
                call.function.arguments.trim().length > 0;
              return (
                <div
                  key={i}
                  className={`mb-1 p-1 border rounded text-xs relative ${
                    isTool
                      ? "bg-green-100 border-green-200"
                      : "bg-yellow-100 border-yellow-200"
                  }`}
                >
                  {/* Copy button for tool call arguments */}
                  {hasToolCallArguments && (
                    <div className="absolute top-1 right-1">
                      <Tooltip
                        content={
                          toolCallCopySuccess[i]
                            ? "Copied!"
                            : "Copy tool call arguments"
                        }
                        position="top"
                      >
                        <Button
                          onClick={() =>
                            handleToolCallCopy(i, call.function.arguments)
                          }
                          size="sm"
                          variant="secondary"
                          className={`p-0.5 h-5 text-[10px] opacity-60 hover:opacity-100 transition-opacity cursor-pointer ${
                            isTool
                              ? "text-green-600 hover:bg-green-200"
                              : "text-yellow-600 hover:bg-yellow-200"
                          }`}
                        >
                          Copy
                        </Button>
                      </Tooltip>
                    </div>
                  )}
                  <div
                    className={`font-semibold mb-0.5 text-xs ${
                      hasToolCallArguments ? "pr-8" : ""
                    } ${isTool ? "text-green-800" : "text-yellow-800"}`}
                  >
                    {call.function.name}
                  </div>
                  <div
                    className={`font-mono text-xs break-all overflow-hidden ${
                      isTool ? "text-green-700" : "text-yellow-700"
                    }`}
                  >
                    {call.function.arguments}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {hasFunctionCall && message.function_call && (
          <div
            className={`mt-2 pt-1 border-t ${
              isTool ? "border-green-200" : "border-yellow-200"
            }`}
          >
            <div
              className={`font-semibold text-xs mb-0.5 ${
                isTool ? "text-green-700" : "text-yellow-700"
              }`}
            >
              Function Call:
            </div>
            <div
              className={`p-1 border rounded text-xs relative ${
                isTool
                  ? "bg-green-100 border-green-200"
                  : "bg-yellow-100 border-yellow-200"
              }`}
            >
              {/* Copy button for function call arguments */}
              {message.function_call.arguments.trim().length > 0 && (
                <div className="absolute top-1 right-1">
                  <Tooltip
                    content={
                      toolCallCopySuccess[-1]
                        ? "Copied!"
                        : "Copy function call arguments"
                    }
                    position="top"
                  >
                    <Button
                      onClick={() =>
                        handleToolCallCopy(-1, message.function_call!.arguments)
                      }
                      size="sm"
                      variant="secondary"
                      className={`p-0.5 h-5 text-[10px] opacity-60 hover:opacity-100 transition-opacity cursor-pointer ${
                        isTool
                          ? "text-green-600 hover:bg-green-200"
                          : "text-yellow-600 hover:bg-yellow-200"
                      }`}
                    >
                      Copy
                    </Button>
                  </Tooltip>
                </div>
              )}
              <div
                className={`font-semibold mb-0.5 text-xs ${
                  message.function_call.arguments.trim().length > 0
                    ? "pr-8"
                    : ""
                } ${isTool ? "text-green-800" : "text-yellow-800"}`}
              >
                {message.function_call.name}
              </div>
              <div
                className={`font-mono text-xs break-all overflow-hidden ${
                  isTool ? "text-green-700" : "text-yellow-700"
                }`}
              >
                {message.function_call.arguments}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
