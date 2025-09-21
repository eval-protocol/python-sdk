import { makeAutoObservable } from "mobx";
import type { Message } from "../types/eval-protocol";
import type { ToolResult } from "../services/AgentService";

export class ChatState {
  messages: Message[] = [];
  isLoading = false;
  error: string | null = null;

  constructor() {
    makeAutoObservable(this);

    // Add welcome message on initialization
    this.addWelcomeMessage();
  }

  addWelcomeMessage() {
    const welcomeMessage: Message = {
      role: "assistant",
      content:
        "Hello! I'm your evaluation analysis assistant. I can help you analyze your evaluation data, find trends, compare models, and discover insights. What would you like to explore?",
    };
    this.messages.push(welcomeMessage);
  }

  // Add a new message
  addMessage(message: Message) {
    this.messages.push(message);
  }

  // Update the last message (useful for streaming responses)
  updateLastMessage(updates: Partial<Message>) {
    if (this.messages.length > 0) {
      const lastMessage = this.messages[this.messages.length - 1];
      Object.assign(lastMessage, updates);
    }
  }

  // Add tool results as separate tool messages
  addToolResultsToLastMessage(toolResults: ToolResult[]) {
    toolResults.forEach((toolResult) => {
      const toolMessage: Message = {
        role: "tool",
        content: toolResult.success
          ? JSON.stringify(toolResult.data)
          : `Error: ${toolResult.error}`,
        tool_call_id: toolResult.id,
      };
      this.messages.push(toolMessage);
    });
  }

  // Clear all messages
  clearMessages() {
    this.messages = [];
  }

  // Set loading state
  setLoading(loading: boolean) {
    this.isLoading = loading;
  }

  // Set error state
  setError(error: string | null) {
    this.error = error;
  }

  // Get messages for a specific conversation (if we add conversation support later)
  getMessagesForConversation(_conversationId?: string): Message[] {
    // For now, return all messages
    // Later we can filter by conversationId
    return this.messages;
  }

  // Get the last message
  getLastMessage(): Message | undefined {
    return this.messages[this.messages.length - 1];
  }

  // Check if the last message has pending tool calls
  hasPendingToolCalls(): boolean {
    const lastMessage = this.getLastMessage();
    return !!(lastMessage?.tool_calls && lastMessage.tool_calls.length > 0);
  }
}
