import type { Message } from "../types/eval-protocol";
import { useState } from "react";
import { BubbleContainer } from "./BubbleContainer";

export const MessageBubble = ({ message }: { message: Message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isTool = message.role === "tool";

  // Check for tool calls and results
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;

  // Get the message content as a string
  const reasoning = (message as any).reasoning_content as string | undefined;
  const getMessageContent = () => {
    if (typeof message.content === "string") {
      return message.content;
    } else if (Array.isArray(message.content)) {
      return message.content
        .map((part) =>
          part.type === "text" ? part.text : JSON.stringify(part)
        )
        .join("");
    } else {
      return JSON.stringify(message.content);
    }
  };

  const messageContent = getMessageContent();
  const isLongMessage = messageContent.length > 200; // Threshold for considering a message "long"
  const displayContent =
    isLongMessage && !isExpanded
      ? messageContent.substring(0, 200) + "..."
      : messageContent;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(messageContent);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error("Failed to copy message:", err);
    }
  };

  return (
    <BubbleContainer
      role={message.role as "user" | "assistant" | "system" | "tool"}
      onCopy={handleCopy}
      copySuccess={copySuccess}
    >
      <div className="whitespace-pre-wrap break-words overflow-hidden text-xs">
        {displayContent}
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
      {hasToolCalls && (
        <div className="mt-2 pt-1 border-t border-gray-200">
          <div className="font-semibold text-xs mb-0.5 text-gray-700">
            Tool Calls:
          </div>
          {message.tool_calls!.map((call: any, i: number) => (
            <div
              key={i}
              className="mb-1 p-1 border rounded text-xs bg-gray-100 border-gray-200"
            >
              <div className="font-semibold mb-0.5 text-xs text-gray-800">
                🔧 {call.function.name}
              </div>
              <div className="font-mono text-xs break-words overflow-hidden text-gray-700">
                {call.function.arguments}
              </div>
            </div>
          ))}
        </div>
      )}
      {hasFunctionCall && (
        <div className="mt-2 pt-1 border-t border-gray-200">
          <div className="font-semibold text-xs mb-0.5 text-gray-700">
            Function Call:
          </div>
          <div className="p-1 border rounded text-xs bg-gray-100 border-gray-200">
            <div className="font-semibold mb-0.5 text-xs text-gray-800">
              🔧 {message.function_call!.name}
            </div>
            <div className="font-mono text-xs break-words overflow-hidden text-gray-700">
              {message.function_call!.arguments}
            </div>
          </div>
        </div>
      )}
    </BubbleContainer>
  );
};
