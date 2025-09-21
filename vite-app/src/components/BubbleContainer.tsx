import type { ReactNode } from "react";
import Button from "./Button";
import { Tooltip } from "./Tooltip";

interface BubbleContainerProps {
  role: "user" | "assistant" | "system" | "tool" | "thinking";
  children: ReactNode;
  onCopy?: () => void;
  copySuccess?: boolean;
  showCopyButton?: boolean;
}

export const BubbleContainer = ({
  role,
  children,
  onCopy,
  copySuccess = false,
  showCopyButton = true,
}: BubbleContainerProps) => {
  const isUser = role === "user";
  const isSystem = role === "system";
  const isTool = role === "tool";
  const isAssistant = role === "assistant";
  const isThinking = role === "thinking";

  const handleCopy = async () => {
    if (onCopy) {
      await onCopy();
    }
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-1`}>
      <div
        className={`max-w-sm lg:max-w-md xl:max-w-lg px-2 py-1 border text-xs relative overflow-scroll ${
          isUser
            ? "bg-blue-50 border-blue-200 text-blue-900"
            : isAssistant
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : isSystem
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : isTool
            ? "bg-green-50 border-green-200 text-green-900"
            : isThinking
            ? "bg-gray-100 border-gray-200 text-gray-800"
            : "bg-yellow-50 border-yellow-200 text-yellow-900"
        }`}
      >
        {/* Copy button positioned in top-right corner */}
        {showCopyButton && onCopy && (
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
                    : isThinking
                    ? "text-gray-600 hover:bg-gray-100"
                    : "text-yellow-600 hover:bg-yellow-100"
                }`}
              >
                Copy
              </Button>
            </Tooltip>
          </div>
        )}

        <div className="font-semibold text-xs mb-0.5 capitalize pr-8">
          {role}
        </div>
        {children}
      </div>
    </div>
  );
};
