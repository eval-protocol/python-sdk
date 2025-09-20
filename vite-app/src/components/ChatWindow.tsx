import { useState } from "react";
import { ChatMessages } from "./ChatMessages";
import Textarea from "./Textarea";
import type { Message } from "../types/eval-protocol";

interface ChatWindowProps {
  className?: string;
}

export const ChatWindow = ({ className = "" }: ChatWindowProps) => {
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");

  const handleSendMessage = () => {
    if (!chatInput.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: chatInput.trim(),
    };

    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput("");

    // Simulate AI response (you can replace this with actual AI integration)
    setTimeout(() => {
      const aiMessage: Message = {
        role: "assistant",
        content: `I received your message: "${userMessage.content}". This is a placeholder response. You can integrate with your AI service here.`,
      };
      setChatMessages((prev) => [...prev, aiMessage]);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className={`flex flex-col ${className}`}>
      <div className="bg-white border border-gray-200 flex flex-col h-[calc(100vh-5rem)]">
        {/* Chat header - following Dashboard pattern */}
        <div className="px-3 py-2 border-b border-gray-200 bg-gray-50">
          <h3 className="text-sm font-medium text-gray-900">AI Assistant</h3>
        </div>

        {/* Chat messages */}
        <ChatMessages messages={chatMessages} />

        {/* Chat input - following Dashboard pattern */}
        <div className="p-3 border-t border-gray-200">
          <Textarea
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message and press Enter to send..."
            className="w-full resize-none"
            size="sm"
            rows={3}
          />
        </div>
      </div>
    </div>
  );
};
