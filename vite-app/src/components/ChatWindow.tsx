import { useState } from "react";
import { observer } from "mobx-react";
import { ChatMessages } from "./ChatMessages";
import Textarea from "./Textarea";
import Button from "./Button";
import { AgentService } from "../services/AgentService";
import type { Message } from "../types/eval-protocol";
import { ChatState } from "../stores/ChatState";

interface ChatWindowProps {
  className?: string;
}

// Create singleton instances at module level
const agentService = new AgentService();
const chatState = new ChatState();

export const ChatWindow = observer(({ className = "" }: ChatWindowProps) => {
  const [chatInput, setChatInput] = useState("");

  const processMessage = async (message: string) => {
    // Add user message
    const userMessage: Message = {
      role: "user",
      content: message,
    };
    chatState.addMessage(userMessage);
    chatState.setLoading(true);
    chatState.setError(null);

    try {
      // For now, simulate AI response with tool calls
      // In a real implementation, you'd call an AI service here
      await simulateAIResponse(message);
    } catch (error) {
      chatState.setError(
        error instanceof Error ? error.message : "Unknown error"
      );
    } finally {
      chatState.setLoading(false);
    }
  };

  const simulateAIResponse = async (userMessage: string) => {
    // Simulate AI thinking time
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Generate a simple response based on the message
    let response = "I understand you want to analyze your evaluation data. ";
    let toolCalls = [];

    // Simple keyword-based tool call generation
    if (
      userMessage.toLowerCase().includes("failed") ||
      userMessage.toLowerCase().includes("error")
    ) {
      response += "Let me find failed evaluations for you.";
      toolCalls = [
        {
          id: agentService.generateToolCallId(),
          name: "analyzeData",
          parameters: {
            filters: [
              {
                field: "evaluation_result.score",
                operator: "<",
                value: "0.5",
              },
            ],
            visualizationType: "table",
          },
        },
      ];
    } else if (
      userMessage.toLowerCase().includes("model") ||
      userMessage.toLowerCase().includes("compare")
    ) {
      response += "Let me compare model performance for you.";
      toolCalls = [
        {
          id: agentService.generateToolCallId(),
          name: "analyzeData",
          parameters: {
            groupBy: ["input_metadata.completion_params.model"],
            aggregations: [
              {
                field: "evaluation_result.score",
                operation: "avg",
                alias: "average_score",
              },
            ],
            visualizationType: "chart",
          },
        },
      ];
    } else if (
      userMessage.toLowerCase().includes("trend") ||
      userMessage.toLowerCase().includes("time")
    ) {
      response += "Let me analyze trends over time for you.";
      toolCalls = [
        {
          id: agentService.generateToolCallId(),
          name: "analyzeData",
          parameters: {
            groupBy: ["created_at"],
            aggregations: [
              {
                field: "evaluation_result.score",
                operation: "avg",
                alias: "average_score",
              },
            ],
            visualizationType: "chart",
          },
        },
      ];
    } else {
      response += "Let me show you a general overview of your data.";
      toolCalls = [
        {
          id: agentService.generateToolCallId(),
          name: "analyzeData",
          parameters: {
            limit: 10,
            visualizationType: "table",
          },
        },
      ];
    }

    // Add AI message
    const aiMessage: Message = {
      role: "assistant",
      content: response,
      tool_calls: toolCalls.map((tc) => ({
        id: tc.id,
        type: "function" as const,
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.parameters),
        },
      })),
    };
    chatState.addMessage(aiMessage);

    // Execute tool calls
    if (toolCalls.length > 0) {
      const toolResults = [];
      for (const toolCall of toolCalls) {
        const result = await agentService.executeToolCall(toolCall);
        toolResults.push(result);
      }
      chatState.addToolResultsToLastMessage(toolResults);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const message = chatInput.trim();
    setChatInput("");
    await processMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setChatInput(suggestion);
  };

  const handleClearChat = () => {
    chatState.clearMessages();
    // Re-add welcome message
    chatState.addWelcomeMessage();
  };

  const toolSuggestions = agentService.getToolSuggestions();

  return (
    <div className={`flex flex-col ${className}`}>
      <div className="bg-white border border-gray-200 flex flex-col h-[calc(100vh-5rem)]">
        {/* Chat header - following Dashboard pattern */}
        <div className="px-3 py-2 border-b border-gray-200 bg-gray-50">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-medium text-gray-900">AI Assistant</h3>
            <Button
              onClick={handleClearChat}
              size="sm"
              variant="secondary"
              disabled={chatState.messages.length <= 1}
            >
              Clear
            </Button>
          </div>
        </div>

        {/* Chat messages */}
        <ChatMessages
          messages={chatState.messages}
          isLoading={chatState.isLoading}
        />

        {/* Tool suggestions */}
        <div className="p-3 border-t border-gray-200 bg-gray-50">
          <div className="flex flex-wrap gap-2 mb-3">
            <span className="text-xs text-gray-500 mr-2">Try asking:</span>
            {toolSuggestions.map((suggestion, index) => (
              <Button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                size="sm"
                variant="secondary"
                disabled={chatState.isLoading}
                className="text-xs px-2 py-1 h-6"
              >
                {suggestion}
              </Button>
            ))}
          </div>
        </div>

        {/* Chat input */}
        <div className="p-3 border-t border-gray-200 bg-white">
          <Textarea
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your evaluation data... (Press Enter to send)"
            className="w-full resize-none"
            size="sm"
            rows={3}
            disabled={chatState.isLoading}
          />
        </div>
      </div>
    </div>
  );
});
