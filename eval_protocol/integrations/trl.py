"""
Adapters for integrating Eval Protocol reward functions with TRL (Transformer Reinforcement Learning) trainers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from eval_protocol.models import Message

logger = logging.getLogger(__name__)


def create_trl_adapter(
    reward_fn: Callable,
    dataset_to_reward_kwargs_map: Dict[str, str],
    static_reward_kwargs: Optional[Dict[str, Any]] = None,
    user_message_fn: Optional[Callable[[Any], str]] = None,  # Function to construct user message content
    assistant_message_fn: Optional[Callable[[Any], str]] = None,  # Function to construct assistant message content
) -> Callable[[List[Any], List[str]], List[float]]:
    """
    Creates an adapter function compatible with TRL trainers (e.g., GRPOTrainer, PPOTrainer)
    from an Eval Protocol reward function.

    The TRL trainer expects a reward function with the signature:
    (prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]
    where **kwargs contains other columns from the HuggingFace dataset.

    Args:
        reward_fn: The Eval Protocol reward function to adapt. This function should
                   already be decorated with @reward_function or follow its
                   input/output conventions (takes List[Message] or List[Dict],
                   returns Dict with a 'score' key).
        dataset_to_reward_kwargs_map: A dictionary mapping dataset column names
                                      (which appear as keys in **kwargs passed by TRL)
                                      to the parameter names of the `reward_fn`.
                                      Example: {"test_cases_column": "test_cases_param"}
                                      This tells the adapter to take the data from
                                      kwargs['test_cases_column'] and pass it as
                                      the `test_cases_param` argument to `reward_fn`.
        static_reward_kwargs: A dictionary of static keyword arguments that will be
                              passed to `reward_fn` for every sample.
                              Example: {"language": "python", "timeout": 10}
        user_message_fn: An optional function that takes a prompt string and returns
                         the content for the user message. If None, the prompt itself
                         is used as content.
        assistant_message_fn: An optional function that takes a completion string and
                              returns the content for the assistant message. If None,
                              the completion itself is used as content.


    Returns:
        An adapter function that can be passed to TRL trainers.
    """
    if static_reward_kwargs is None:
        static_reward_kwargs = {}

    def trl_reward_pipeline(
        prompts: List[Any],  # Changed from List[str] to List[Any]
        completions: Optional[List[str]] = None,
        **kwargs: Any,  # Contains other dataset columns, e.g., kwargs['test_cases']
    ) -> List[float]:
        """
        This is the actual function TRL will call.

        Note: completions parameter is optional to handle cases where prompts already
        contain complete conversations.
        """
        scores: List[float] = []
        num_samples = len(prompts)

        # If completions is None, assume prompts contains complete conversations
        if completions is None:
            completions = [""] * num_samples

        if not (len(completions) == num_samples):
            logger.warning(
                f"Mismatch in lengths of prompts ({num_samples}) and "
                f"completions ({len(completions)}). Using min length."
            )
            num_samples = min(num_samples, len(completions))

        # Pre-extract data for all samples from kwargs based on the map
        # This makes it easier to access per-sample data in the loop
        mapped_kwargs_data: Dict[str, List[Any]] = {}
        for (
            dataset_col_name,
            reward_fn_param_name,
        ) in dataset_to_reward_kwargs_map.items():
            if dataset_col_name not in kwargs:
                logger.warning(
                    f"Dataset column '{dataset_col_name}' (mapped to reward_fn param "
                    f"'{reward_fn_param_name}') not found in TRL kwargs. "
                    f"Reward function will receive None for this parameter for all samples."
                )
                # Ensure the key exists in mapped_kwargs_data with a list of Nones
                mapped_kwargs_data[reward_fn_param_name] = [None] * num_samples
            else:
                # Ensure the data from TRL kwargs is a list of the correct length
                data_list = kwargs[dataset_col_name]
                if not isinstance(data_list, list) or len(data_list) != num_samples:
                    logger.error(
                        f"Data for dataset column '{dataset_col_name}' is not a list of "
                        f"length {num_samples}. Received: {data_list}. "
                        f"Reward function will receive None for this parameter for all samples."
                    )
                    mapped_kwargs_data[reward_fn_param_name] = [None] * num_samples
                else:
                    mapped_kwargs_data[reward_fn_param_name] = data_list

        for i in range(num_samples):
            current_prompt_item: Any = prompts[i]
            current_completion: str = completions[i]

            # Construct messages
            # If user_message_fn is provided, it's responsible for converting current_prompt_item to string content.
            # If not, and current_prompt_item is not a string, this might error or behave unexpectedly.
            # Default behavior: assume current_prompt_item is a string if user_message_fn is None.
            user_content = user_message_fn(current_prompt_item) if user_message_fn else str(current_prompt_item)

            # Default extraction for assistant_content if current_completion is not a simple string
            final_assistant_str_content = ""
            if assistant_message_fn:
                final_assistant_str_content = assistant_message_fn(current_completion)
            elif isinstance(current_completion, str):
                final_assistant_str_content = current_completion
            elif (
                isinstance(current_completion, list)
                and len(current_completion) == 1
                and isinstance(current_completion[0], dict)
                and "content" in current_completion[0]
                and isinstance(current_completion[0].get("content"), str)
            ):
                # Handles cases like [{'role':'assistant', 'content':'actual_text'}]
                final_assistant_str_content = current_completion[0]["content"]
            else:
                # Fallback if current_completion is an unexpected type
                logger.warning(
                    f"Completion for assistant message was not a string or expected list/dict structure: {current_completion}. Using str()."
                )
                final_assistant_str_content = str(current_completion)

            # Ensure messages_for_reward is typed as List[Message] as per EvaluateFunction protocol
            messages_for_reward: List[Message] = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=final_assistant_str_content),
            ]

            # Prepare kwargs for the specific reward_fn call for this sample
            current_dynamic_kwargs: Dict[str, Any] = {}
            for reward_fn_param_name, data_list_for_param in mapped_kwargs_data.items():
                # data_list_for_param is already ensured to be a list of Nones or actual data
                current_dynamic_kwargs[reward_fn_param_name] = data_list_for_param[i]

            # Combine static and dynamic kwargs
            final_reward_fn_kwargs = {**static_reward_kwargs, **current_dynamic_kwargs}

            try:
                # reward_fn is expected to be decorated with @reward_function,
                # so it handles Message object creation internally if dicts are passed,
                # and returns a dict.
                reward_output_dict: Dict[str, Any] = reward_fn(messages=messages_for_reward, **final_reward_fn_kwargs)

                score = reward_output_dict.get("score")
                if score is None:
                    logger.warning(
                        f"Sample {i}: 'score' key not found in reward_output_dict or is None. "
                        f"Output: {reward_output_dict}. Assigning 0.0."
                    )
                    scores.append(0.0)
                else:
                    scores.append(float(score))

            except Exception as e:
                logger.error(
                    f"Error calling reward_fn for sample {i} (prompt: '{str(current_prompt_item)[:50]}...'): {e}",
                    exc_info=True,
                )
                scores.append(0.0)  # Assign 0 score on error

        if scores:
            logger.debug(
                f"Batch rewards calculated by TRL adapter. Count: {len(scores)}, "
                f"Min: {min(scores)}, Max: {max(scores)}, Avg: {sum(scores) / len(scores):.2f}"
            )
        return scores

    return trl_reward_pipeline
