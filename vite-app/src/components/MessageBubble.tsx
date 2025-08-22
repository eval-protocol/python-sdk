import type { Message } from "../types/eval-protocol";
import { useState } from "react";
import Button from "./Button";
import { Tooltip } from "./Tooltip";

export const MessageBubble = ({ message }: { message: Message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isTool = message.role === "tool";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;

  // Get the message content as a string
  const getMessageContent = () => {
    if (typeof message.content === "string") {
      return message.content;
    } else if (Array.isArray(message.content)) {
      return message.content
        .map((part, i) =>
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

        <div className="font-semibold text-xs mb-0.5 capitalize pr-8">
          {message.role}
        </div>
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
            {message.tool_calls.map((call, i) => (
              <div
                key={i}
                className={`mb-1 p-1 border rounded text-xs ${
                  isTool
                    ? "bg-green-100 border-green-200"
                    : "bg-yellow-100 border-yellow-200"
                }`}
              >
                <div
                  className={`font-semibold mb-0.5 text-xs ${
                    isTool ? "text-green-800" : "text-yellow-800"
                  }`}
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
            ))}
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
              className={`p-1 border rounded text-xs ${
                isTool
                  ? "bg-green-100 border-green-200"
                  : "bg-yellow-100 border-yellow-200"
              }`}
            >
              <div
                className={`font-semibold mb-0.5 text-xs ${
                  isTool ? "text-green-800" : "text-yellow-800"
                }`}
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
