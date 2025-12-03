"""
VLLMPolicy - Policy for TRL's VLLMClient or colocated vLLM LLM.

Thin adapter that turns Eval Protocol-style message lists into a single prompt,
then calls either:

- TRL's VLLMClient (server mode), or
- a colocated vLLM LLM instance (SamplingParams mode).
"""

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class VLLMPolicy:
    """
    Policy that uses TRL's VLLMClient for generation.

    This is designed to work with `trl vllm-serve` which provides
    custom /generate/ and /chat/ endpoints.
    """

    def __init__(
        self,
        vllm_client,  # trainer.vllm_client
        tokenizer=None,  # Optional tokenizer for decoding
        temperature: float = 1.0,
        max_tokens: int = 100,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize VLLMPolicy.

        Args:
            vllm_client: TRL's VLLMClient instance (from trainer.vllm_client)
            tokenizer: Optional tokenizer for decoding token IDs to text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
            top_k: Top-k sampling
            **kwargs: Additional generation parameters
        """
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.kwargs = kwargs

    async def _make_llm_call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Make LLM call using TRL's VLLMClient or a colocated vLLM LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Not used (for compatibility)

        Returns:
            OpenAI-compatible response dict
        """
        # Apply chat template to convert messages to a prompt string
        if self.tokenizer is not None:
            try:
                # Use tokenizer's chat template
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                logger.debug(
                    "[VLLMPolicy] Chat template applied for %d messages (prompt length=%d)",
                    len(messages),
                    len(prompt_text),
                )
            except Exception as e:
                logger.warning(
                    "[VLLMPolicy] Failed to apply chat template: %s",
                    e,
                    exc_info=True,
                )
                # Fallback: simple concatenation (defensive .get access)
                prompt_text = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)
        else:
            # No tokenizer: simple concatenation
            prompt_text = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)

        # Check if vllm_client is VLLMClient (server mode) or LLM (colocate mode)
        is_llm_object = hasattr(self.vllm_client, "llm_engine")  # LLM has llm_engine

        if is_llm_object:
            # Colocate mode: use SamplingParams
            logger.debug("[VLLMPolicy] Using vLLM LLM (colocate mode) with SamplingParams")
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                n=1,
            )

            logger.debug("[VLLMPolicy] Calling LLM.generate()")
            outputs = self.vllm_client.generate([prompt_text], sampling_params=sampling_params, use_tqdm=False)

            # Extract from vLLM output format
            output = outputs[0]
            prompt_ids = output.prompt_token_ids
            completion_ids = output.outputs[0].token_ids
            response = {
                "prompt_ids": [prompt_ids],
                "completion_ids": [completion_ids],
            }
        else:
            # Server mode: use VLLMClient with kwargs
            logger.debug("[VLLMPolicy] Using VLLMClient (server mode)")
            vllm_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "n": 1,
            }
            vllm_params.update(self.kwargs)

            logger.debug("[VLLMPolicy] Calling vllm_client.generate()")
            response = self.vllm_client.generate(
                prompts=[prompt_text],
                **vllm_params,
            )

        # Extract first result
        prompt_ids = response["prompt_ids"][0]
        completion_ids = response["completion_ids"][0]

        # Decode completion text if tokenizer available
        if self.tokenizer is not None:
            try:
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                logger.debug(
                    "[VLLMPolicy] Generation result: prompt_tokens=%d, completion_tokens=%d, completion_chars=%d",
                    len(prompt_ids),
                    len(completion_ids),
                    len(completion_text),
                )
            except Exception as e:
                logger.warning(
                    "[VLLMPolicy] Failed to decode completion: %s",
                    e,
                    exc_info=True,
                )
                completion_text = f"<decoded_error:{len(completion_ids)}_tokens>"
        else:
            # Fallback: just indicate number of tokens
            completion_text = f"<{len(completion_ids)}_tokens>"

        # Convert to OpenAI-compatible format for compatibility with OpenEnvRolloutProcessor
        # Also include raw token IDs for TRL integration (avoids double encoding)
        return {
            "choices": [
                {
                    "message": {
                        "content": completion_text,
                        "role": "assistant",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
            # Include raw token IDs for TRL (avoids re-encoding)
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        }
