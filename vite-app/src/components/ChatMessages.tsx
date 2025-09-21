import { useRef, useEffect } from "react";
import type { Message } from "../types/eval-protocol";
import { MessageBubble } from "./MessageBubble";
import { ThinkingBubble } from "./ThinkingBubble";

interface ChatMessagesProps {
  messages: Message[];
  isLoading?: boolean;
}

export const ChatMessages = ({
  messages,
  isLoading = false,
}: ChatMessagesProps) => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const prevMessagesLengthRef = useRef(0);

  // Auto-scroll to bottom when new messages come in
  useEffect(() => {
    // On first render, just set the initial length without scrolling
    if (prevMessagesLengthRef.current === 0) {
      prevMessagesLengthRef.current = messages.length;
      return;
    }

    // Only scroll if we have messages and the number of messages has increased
    // This prevents scrolling on initial mount or when messages are removed
    if (
      messages.length > 0 &&
      messages.length > prevMessagesLengthRef.current
    ) {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTo({
          top: scrollContainerRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }
    // Update the previous length for the next comparison
    prevMessagesLengthRef.current = messages.length;
  }, [messages]);

  return (
    <div
      ref={scrollContainerRef}
      className="flex-1 overflow-y-auto p-3 space-y-2"
    >
      {messages.length === 0 ? (
        <div className="flex items-center justify-center h-full text-gray-500">
          <div className="text-center">
            <div className="text-lg mb-2">🤖</div>
            <div className="text-sm">
              Start a conversation with the AI assistant
            </div>
            <div className="text-xs mt-1">
              Ask about your evaluation data, trends, or insights
            </div>
          </div>
        </div>
      ) : (
        messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))
      )}

      {isLoading && <ThinkingBubble />}
    </div>
  );
};
