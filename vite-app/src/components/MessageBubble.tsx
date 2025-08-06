import type { Message } from "../types/eval-protocol";

export const MessageBubble = ({ message }: { message: Message }) => {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
      <div
        className={`max-w-xs lg:max-w-md px-3 py-2 border text-sm ${
          isUser
            ? "bg-blue-50 border-blue-200 text-blue-900"
            : isSystem
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : hasToolCalls || hasFunctionCall
            ? "bg-orange-50 border-orange-200 text-orange-900"
            : "bg-white border-gray-200 text-gray-900"
        }`}
      >
        <div className="font-semibold text-xs mb-1 capitalize">
          {message.role}
          {(hasToolCalls || hasFunctionCall) && (
            <span className="ml-1 text-orange-600">• Tool Call</span>
          )}
        </div>
        <div className="whitespace-pre-wrap">
          {typeof message.content === "string"
            ? message.content
            : Array.isArray(message.content)
            ? message.content
                .map((part, i) =>
                  part.type === "text" ? part.text : JSON.stringify(part)
                )
                .join("")
            : JSON.stringify(message.content)}
        </div>
        {hasToolCalls && message.tool_calls && (
          <div className="mt-3 pt-2 border-t border-orange-200">
            <div className="font-semibold text-xs text-orange-700 mb-1">
              Tool Calls:
            </div>
            {message.tool_calls.map((call, i) => (
              <div
                key={i}
                className="mb-2 p-2 bg-orange-100 border border-orange-200 rounded text-xs"
              >
                <div className="font-semibold text-orange-800 mb-1">
                  {call.function.name}
                </div>
                <div className="text-orange-700 font-mono text-xs">
                  {call.function.arguments}
                </div>
              </div>
            ))}
          </div>
        )}
        {hasFunctionCall && message.function_call && (
          <div className="mt-3 pt-2 border-t border-orange-200">
            <div className="font-semibold text-xs text-orange-700 mb-1">
              Function Call:
            </div>
            <div className="p-2 bg-orange-100 border border-orange-200 rounded text-xs">
              <div className="font-semibold text-orange-800 mb-1">
                {message.function_call.name}
              </div>
              <div className="text-orange-700 font-mono text-xs">
                {message.function_call.arguments}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
