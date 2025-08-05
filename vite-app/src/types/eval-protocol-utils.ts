import type { EvaluationRow, Message } from './eval-protocol';

/**
 * Utility functions for working with EvaluationRow data
 * These mirror the methods from the Python EvaluationRow class
 */

export const evalRowUtils = {
  /**
   * Returns True if this represents a trajectory evaluation (has step_outputs),
   * False if it represents a single turn evaluation.
   */
  isTrajectoryEvaluation: (row: EvaluationRow): boolean => {
    return (
      row.evaluation_result !== undefined &&
      row.evaluation_result.step_outputs !== undefined &&
      row.evaluation_result.step_outputs.length > 0
    );
  },

  /**
   * Returns the number of messages in the conversation.
   */
  getConversationLength: (row: EvaluationRow): number => {
    return row.messages.length;
  },

  /**
   * Returns the system message from the conversation. Returns empty Message if none found.
   */
  getSystemMessage: (row: EvaluationRow): Message => {
    const systemMessages = row.messages.filter(msg => msg.role === 'system');
    if (systemMessages.length === 0) {
      return { role: 'system', content: '' };
    }
    return systemMessages[0];
  },

  /**
   * Returns only the assistant messages from the conversation.
   */
  getAssistantMessages: (row: EvaluationRow): Message[] => {
    return row.messages.filter(msg => msg.role === 'assistant');
  },

  /**
   * Returns only the user messages from the conversation.
   */
  getUserMessages: (row: EvaluationRow): Message[] => {
    return row.messages.filter(msg => msg.role === 'user');
  },

  /**
   * Helper method to get a specific value from input_metadata.
   */
  getInputMetadata: (row: EvaluationRow, key: string, defaultValue?: any): any => {
    if (!row.input_metadata) {
      return defaultValue;
    }
    return (row.input_metadata as any)[key] ?? defaultValue;
  },

  /**
   * Get number of steps from control_plane_step data.
   */
  getSteps: (row: EvaluationRow): number => {
    return row.messages.filter(msg => msg.control_plane_step).length;
  },

  /**
   * Get total reward from control_plane_step data.
   */
  getTotalReward: (row: EvaluationRow): number => {
    const messagesWithControlPlane = row.messages.filter(msg => msg.control_plane_step);
    if (messagesWithControlPlane.length === 0) {
      return 0.0;
    }
    return messagesWithControlPlane.reduce((total, msg) => {
      const reward = (msg.control_plane_step as any)?.reward;
      return total + (typeof reward === 'number' ? reward : 0);
    }, 0.0);
  },

  /**
   * Get termination status from control_plane_step data.
   */
  getTerminated: (row: EvaluationRow): boolean => {
    const messagesWithControlPlane = row.messages.filter(msg => msg.control_plane_step);
    if (messagesWithControlPlane.length === 0) {
      return false;
    }
    return messagesWithControlPlane.some(msg => {
      return (msg.control_plane_step as any)?.terminated === true;
    });
  },

  /**
   * Get termination reason from the final control_plane_step data.
   */
  getTerminationReason: (row: EvaluationRow): string => {
    // Find the last message with control_plane_step that has termination_reason
    for (let i = row.messages.length - 1; i >= 0; i--) {
      const msg = row.messages[i];
      if (msg.control_plane_step && (msg.control_plane_step as any)?.termination_reason) {
        return (msg.control_plane_step as any).termination_reason;
      }
    }
    return 'unknown';
  }
};

/**
 * Utility functions for working with Message data
 */
export const messageUtils = {
  /**
   * Check if a message has tool calls
   */
  hasToolCalls: (message: Message): boolean => {
    return message.tool_calls !== undefined && message.tool_calls.length > 0;
  },

  /**
   * Check if a message has function calls
   */
  hasFunctionCall: (message: Message): boolean => {
    return message.function_call !== undefined;
  },

  /**
   * Get the content as a string, handling both string and array content types
   */
  getContentAsString: (message: Message): string => {
    if (typeof message.content === 'string') {
      return message.content;
    }
    if (Array.isArray(message.content)) {
      return message.content
        .filter(part => part.type === 'text')
        .map(part => part.text)
        .join('');
    }
    return '';
  }
};

/**
 * Utility functions for working with EvaluateResult data
 */
export const evaluateResultUtils = {
  /**
   * Check if the evaluation result has step outputs (trajectory evaluation)
   */
  hasStepOutputs: (result: any): boolean => {
    return result.step_outputs !== undefined && result.step_outputs.length > 0;
  },

  /**
   * Get the total base reward from step outputs
   */
  getTotalBaseReward: (result: any): number => {
    if (!result.step_outputs) {
      return 0.0;
    }
    return result.step_outputs.reduce((total: number, step: any) => {
      return total + (step.base_reward || 0);
    }, 0.0);
  },

  /**
   * Get the number of steps from step outputs
   */
  getStepCount: (result: any): number => {
    return result.step_outputs?.length || 0;
  }
};
