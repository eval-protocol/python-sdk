import type { Message } from "../types/eval-protocol";

export const MessageBubble = ({ message }: { message: Message }) => {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isTool = message.role === "tool";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
      <div
        className={`max-w-sm lg:max-w-md xl:max-w-lg px-3 py-2 border text-sm ${
          isUser
            ? "bg-blue-50 border-blue-200 text-blue-900"
            : isSystem
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : isTool
            ? "bg-green-50 border-green-200 text-green-900"
            : "bg-yellow-50 border-yellow-200 text-yellow-900"
        }`}
      >
        <div className="font-semibold text-xs mb-1 capitalize">
          {message.role}
        </div>
        <div className="whitespace-pre-wrap break-words overflow-hidden">
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
          <div
            className={`mt-3 pt-2 border-t ${
              isTool ? "border-green-200" : "border-yellow-200"
            }`}
          >
            <div
              className={`font-semibold text-xs mb-1 ${
                isTool ? "text-green-700" : "text-yellow-700"
              }`}
            >
              Tool Calls:
            </div>
            {message.tool_calls.map((call, i) => (
              <div
                key={i}
                className={`mb-2 p-2 border rounded text-xs ${
                  isTool
                    ? "bg-green-100 border-green-200"
                    : "bg-yellow-100 border-yellow-200"
                }`}
              >
                <div
                  className={`font-semibold mb-1 ${
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
            className={`mt-3 pt-2 border-t ${
              isTool ? "border-green-200" : "border-yellow-200"
            }`}
          >
            <div
              className={`font-semibold text-xs mb-1 ${
                isTool ? "text-green-700" : "text-yellow-700"
              }`}
            >
              Function Call:
            </div>
            <div
              className={`p-2 border rounded text-xs ${
                isTool
                  ? "bg-green-100 border-green-200"
                  : "bg-yellow-100 border-yellow-200"
              }`}
            >
              <div
                className={`font-semibold mb-1 ${
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
