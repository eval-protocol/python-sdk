"""
User simulators for multi-turn dialogue in filesystem RL scenarios.

This module provides two types of user simulators:
1. UserSimulatorLLM: Uses an LLM to simulate user responses dynamically
2. UserSimulatorScripted: Uses pre-defined scripted responses
"""

import logging
from typing import Dict, List, Optional

# Import LLM client (assuming litellm or similar is available)
try:
    import litellm

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    litellm = None

logger = logging.getLogger(__name__)


class UserSimulatorLLM:
    """
    LLM-driven user simulator inspired by tau-bench.

    Uses a separate LLM to act as the user, guided by specific instructions
    and system prompts to create realistic multi-turn interactions.
    """

    def __init__(self, user_llm_config: Dict, task_instruction: str):
        """
        Initialize the LLM user simulator.

        Args:
            user_llm_config: Configuration for the user LLM (model, provider, etc.)
            task_instruction: The overall goal/instruction for the user
        """
        self.user_llm_config = user_llm_config
        self.task_instruction = task_instruction
        self.conversation_history: List[Dict[str, str]] = []
        self.use_mock = not LLM_AVAILABLE or user_llm_config.get("model") == "mock-model"

        if not LLM_AVAILABLE:
            logger.warning("litellm not available, using mock responses")

        # Build and store system prompt
        system_prompt = self._build_system_prompt(task_instruction)
        self.conversation_history.append({"role": "system", "content": system_prompt})

        logger.info(f"UserSimulatorLLM initialized (mock mode: {self.use_mock})")

    def _build_system_prompt(self, instruction: str) -> str:
        """
        Build the system prompt for the user LLM.

        Args:
            instruction: The task instruction to incorporate

        Returns:
            System prompt string
        """
        return f"""You are a helpful user trying to get an AI agent to perform filesystem operations.
Your overall goal is: "{instruction}"

Instructions for interaction:
- Interact with the agent turn-by-turn in a natural, conversational manner
- If the agent asks for clarification, provide only the necessary information for the current step
- Do not reveal your entire goal at once unless asked directly or it's natural to do so
- If the agent seems to have completed your request correctly, you can respond with "Thank you, that looks correct." or similar and then output "###TASK_SATISFIED###" on a new line
- If the agent makes a mistake or asks something irrelevant, gently guide it back or point out the error
- Your responses should be concise and natural
- Be patient but clear about what you want

Remember: You are the user giving instructions to an AI agent with filesystem capabilities."""

    async def generate_response(self, agent_utterance: Optional[str] = None) -> str:
        """
        Generate a user response using the LLM or mock.

        Args:
            agent_utterance: The agent's previous response (None for initial query)

        Returns:
            User's response text
        """
        # Add agent's utterance to history if provided
        if agent_utterance is not None:
            self.conversation_history.append({"role": "assistant", "content": agent_utterance})

        if self.use_mock:
            return await self._generate_mock_response(agent_utterance)

        # Prepare messages for the user LLM
        messages = self.conversation_history.copy()

        try:
            # Call the user LLM
            response = await litellm.acompletion(
                model=self.user_llm_config.get("model", "gpt-3.5-turbo"),
                messages=messages,
                temperature=self.user_llm_config.get("temperature", 0.7),
                max_tokens=self.user_llm_config.get("max_tokens", 150),
                **{k: v for k, v in self.user_llm_config.items() if k not in ["model", "temperature", "max_tokens"]},
            )

            user_reply = response.choices[0].message.content.strip()

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_reply})

            logger.debug(f"User simulator generated: {user_reply}")
            return user_reply

        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            # Fallback to mock
            return await self._generate_mock_response(agent_utterance)

    async def _generate_mock_response(self, agent_utterance: Optional[str] = None) -> str:
        """Generate mock user responses for testing."""
        turn_count = len([msg for msg in self.conversation_history if msg["role"] == "user"])

        if agent_utterance is None:
            # Initial query
            return self.task_instruction

        # Mock responses based on turn and content
        if turn_count == 0:
            # First response after initial agent message
            if "move" in self.task_instruction.lower():
                return "Yes, please move that file for me."
            elif "copy" in self.task_instruction.lower():
                return "Please go ahead and copy those files."
            elif "create" in self.task_instruction.lower():
                return "Please create that file with the specified content."
            else:
                return "Please proceed with the task."

        elif turn_count == 1:
            # Second response
            if "successfully" in agent_utterance.lower() or "completed" in agent_utterance.lower():
                return "Thank you, that looks correct. ###TASK_SATISFIED###"
            else:
                return "Sounds good, please continue."

        else:
            # Later responses
            return "Thank you, that looks correct. ###TASK_SATISFIED###"


class UserSimulatorScripted:
    """
    Scripted user simulator inspired by BFCL.

    Uses pre-defined user utterances in sequence to create predictable
    multi-turn interactions for testing and evaluation.
    """

    def __init__(self, scripted_turns: List[str]):
        """
        Initialize the scripted user simulator.

        Args:
            scripted_turns: List of pre-defined user utterances
        """
        self.scripted_turns = scripted_turns
        self.current_turn_index = 0

        logger.info(f"UserSimulatorScripted initialized with {len(scripted_turns)} turns")

    async def generate_response(self, agent_utterance: Optional[str] = None) -> str:
        """
        Generate the next scripted user response.

        Args:
            agent_utterance: The agent's previous response (logged but not used)

        Returns:
            Next scripted user message or end signal
        """
        # Log agent utterance for debugging
        if agent_utterance is not None:
            logger.debug(f"Agent said: {agent_utterance}")

        # Check if we have more scripted turns
        if self.current_turn_index < len(self.scripted_turns):
            user_message = self.scripted_turns[self.current_turn_index]
            self.current_turn_index += 1
            logger.debug(f"Scripted user turn {self.current_turn_index}: {user_message}")
            return user_message
        else:
            # No more scripted turns
            logger.debug("No more scripted turns available")
            return "###SCRIPT_END###"


# Utility functions for dialogue management
def is_dialogue_complete(user_response: str) -> bool:
    """
    Check if the dialogue should end based on user response.

    Args:
        user_response: The user's response

    Returns:
        True if dialogue should end
    """
    completion_signals = ["###TASK_SATISFIED###", "###SCRIPT_END###"]
    return any(signal in user_response for signal in completion_signals)


def extract_completion_signal(user_response: str) -> Optional[str]:
    """
    Extract completion signal from user response.

    Args:
        user_response: The user's response

    Returns:
        The completion signal if found, None otherwise
    """
    signals = ["###TASK_SATISFIED###", "###SCRIPT_END###"]
    for signal in signals:
        if signal in user_response:
            return signal
    return None
