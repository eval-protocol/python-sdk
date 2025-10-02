"""
Core evaluation execution pipeline for Eval Protocol.
This module orchestrates dataset loading, model response generation (optional),
and evaluation using specified reward functions.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, cast

import aiohttp
import hydra
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf

from eval_protocol.auth import get_fireworks_api_key
from eval_protocol.generation.cache import ResponseCache
from eval_protocol.generation.clients import FireworksModelClient, ModelClient
from eval_protocol.mcp.clients import IntermediaryMCPClient
from eval_protocol.models import Message
from eval_protocol.utils.module_loader import load_function as load_reward_function
from eval_protocol.utils.packaging_utils import install_requirements

try:
    from datasets import Dataset, DatasetDict  # pyright: ignore[reportAttributeAccessIssue]
except ImportError:
    raise ImportError(
        "The 'datasets' package is required to use this function. "
        "Please install it with 'pip install \"eval-protocol[huggingface]\"'"
    )

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(self, pipeline_cfg: DictConfig):
        self.cfg = pipeline_cfg  # Root config for this pipeline run

        # Initialize components based on config
        logger.info("Initializing EvaluationPipeline...")

        self.model_client: Optional[ModelClient] = None
        if self.cfg.generation.enabled:
            api_key = get_fireworks_api_key()
            if not api_key:
                logger.error("Fireworks API key not found, but generation is enabled.")
                raise ValueError("API key required for Fireworks model client when generation is enabled.")
            self.model_client = FireworksModelClient(
                client_config=self.cfg.generation,
                api_key=api_key,
            )
            logger.info(f"Initialized FireworksModelClient for model: {self.cfg.generation.model_name}")

        self.cache = ResponseCache(self.cfg.generation.cache)
        logger.info("ResponseCache initialized.")

        self.reward_function = load_reward_function(self.cfg.reward.function_path)
        logger.info(f"Loaded reward function from: {self.cfg.reward.function_path}")

        # Install requirements if specified by the decorator
        if hasattr(self.reward_function, "_reward_function_requirements"):
            requirements = getattr(self.reward_function, "_reward_function_requirements")
            if isinstance(requirements, list) and requirements:
                logger.info(f"Found requirements for reward function {self.cfg.reward.function_path}: {requirements}")
                try:
                    # Assuming install_requirements uses the current environment's pip by default
                    install_requirements(requirements_list=requirements)
                    logger.info(f"Successfully processed requirements for {self.cfg.reward.function_path}.")
                except Exception as e:
                    logger.error(
                        f"Failed to install requirements for {self.cfg.reward.function_path}: {e}",
                        exc_info=True,
                    )
                    # Depending on policy, might re-raise or allow continuation if some are optional
                    # For now, log error and continue; pip install errors are already logged by the utility.
                    # If strict, could raise RuntimeError here.
            elif requirements:  # Not a list or empty
                logger.warning(
                    f"_reward_function_requirements for {self.cfg.reward.function_path} is not a non-empty list: {requirements}"
                )

        self.mcp_intermediary_client: Optional[IntermediaryMCPClient] = None
        if self.cfg.get("agent") and self.cfg.agent.get("type") == "mcp_agent":
            if not self.cfg.agent.get("intermediary_server_url"):
                raise ValueError("agent.intermediary_server_url must be configured for mcp_agent type.")
            logger.info("Pipeline configured for mcp_agent. IntermediaryMCPClient will be initialized in run().")

    async def _discover_tools_for_sample(self, sample_id: str, mcp_backend_ref: str) -> List[Dict[str, Any]]:
        """Discover available tools from MCP backend for a sample."""
        discovered_tools = []
        rk_session_id = None

        try:
            backend_requests = [{"backend_name_ref": mcp_backend_ref, "num_instances": 1}]
            assert self.mcp_intermediary_client is not None
            init_response = await self.mcp_intermediary_client.initialize_session(backend_requests)

            if init_response.get("error"):
                raise RuntimeError(
                    f"MCP session for tool discovery failed: {init_response.get('error_details', init_response['error'])}"
                )

            rk_session_id = init_response.get("rk_session_id")
            initialized_backends = init_response.get("initialized_backends", [])

            if not rk_session_id or not initialized_backends:
                raise RuntimeError(f"Malformed init response for tool discovery: {init_response}")

            for backend_info in initialized_backends:
                current_backend_name_ref = backend_info.get("backend_name_ref")
                instances_info = backend_info.get("instances", [])
                if not current_backend_name_ref or not instances_info:
                    continue
                for inst_info_dict in instances_info:
                    current_instance_id = inst_info_dict.get("instance_id")
                    if not current_instance_id:
                        continue
                    assert self.mcp_intermediary_client is not None
                    list_tools_result = await self.mcp_intermediary_client.list_backend_tools(
                        rk_session_id=rk_session_id,
                        instance_id=current_instance_id,
                        backend_name_ref=current_backend_name_ref,
                    )
                    if list_tools_result and list_tools_result.tools:
                        for tool_obj in list_tools_result.tools:
                            discovered_tools.append(tool_obj.model_dump(exclude_none=True))

            logger.info(f"Sample {sample_id}: Discovered {len(discovered_tools)} tools.")

        except Exception as e_tool_discovery:
            logger.error(
                f"Sample {sample_id}: Error during tool discovery: {e_tool_discovery}",
                exc_info=True,
            )
            discovered_tools = []
        finally:
            if rk_session_id and self.mcp_intermediary_client:
                logger.info(f"Sample {sample_id}: Cleaning up tool discovery session '{rk_session_id}'.")
                try:
                    assert self.mcp_intermediary_client is not None
                    await self.mcp_intermediary_client.cleanup_session(rk_session_id)
                except Exception as e_cl:
                    logger.error(
                        f"Error cleaning up discovery session '{rk_session_id}': {e_cl}",
                        exc_info=True,
                    )

        return discovered_tools

    async def _process_n_variants(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        user_query: Optional[str],
        ground_truth_for_eval: Optional[str],
        existing_messages: Optional[List[Dict[str, Any]]],
        http_session: Optional[aiohttp.ClientSession],
        n_variants: int,
        original_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Process a sample to generate N variants of responses."""
        results = []

        for variant_idx in range(n_variants):
            # Create a variant-specific sample ID
            variant_sample_id = f"{sample_id}_v{variant_idx}"

            # Create a modified sample for this variant
            variant_sample = sample.copy()
            variant_sample["id"] = variant_sample_id

            # Temporarily set n=1 to avoid infinite recursion
            original_n = self.cfg.generation.get("n", 1)
            self.cfg.generation.n = 1

            try:
                # Process this variant as a single sample
                variant_result = await self._process_single_sample_internal(
                    sample=variant_sample,
                    http_session=http_session,
                    original_index=original_index,
                )

                if variant_result is not None:
                    # Add variant metadata for batch evaluation compatibility
                    variant_result["request_id"] = sample_id  # Original sample ID for grouping
                    variant_result["response_id"] = variant_idx  # Variant index within the request
                    results.append(variant_result)

            except Exception as e:
                logger.error(
                    f"Error processing variant {variant_idx} for sample {sample_id}: {e}",
                    exc_info=True,
                )
                # Add error result for this variant
                results.append(
                    {
                        "id": variant_sample_id,
                        "request_id": sample_id,
                        "response_id": variant_idx,
                        "error": f"Variant processing failed: {str(e)}",
                        "evaluation_score": 0.0,
                    }
                )
            finally:
                # Restore original n value
                self.cfg.generation.n = original_n

        return results

    async def _execute_standard_generation(
        self,
        sample_id: str,
        user_query: str,
        system_prompt_content: Optional[str],
        http_session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Execute standard LLM generation without agent capabilities."""
        current_messages_for_rollout = []
        if system_prompt_content:
            current_messages_for_rollout.append({"role": "system", "content": system_prompt_content})
        current_messages_for_rollout.append({"role": "user", "content": user_query})
        assert self.model_client is not None, "at this point model client needs to be initialized"

        generation_output_std = await self.model_client.generate(
            messages=current_messages_for_rollout,
            session=http_session,
            tools=None,  # No tools for non-agent
        )
        final_assistant_output_for_log = generation_output_std.content

        if not final_assistant_output_for_log:
            logger.warning(f"Sample {sample_id}: Standard generation resulted in no content.")
            final_assistant_output_for_log = "LLM provided no content."

        # Cache standard generation if applicable
        if (
            final_assistant_output_for_log
            and self.cfg.generation.cache.enabled
            and self.model_client.temperature == 0.0
        ):
            self.cache.put(
                sample_id=sample_id,
                system_prompt=system_prompt_content,
                user_query=user_query,
                model_name=self.model_client.model_name,
                temperature=self.model_client.temperature,
                response=final_assistant_output_for_log,
                top_p=self.model_client.top_p,
                top_k=self.model_client.top_k,
                min_p=self.model_client.min_p,
                max_tokens=self.model_client.max_tokens,
                reasoning_effort=self.model_client.reasoning_effort,
            )

        return {
            "success": True,
            "final_assistant_output": final_assistant_output_for_log,
            "conversation_history": current_messages_for_rollout
            + [{"role": "assistant", "content": final_assistant_output_for_log}],
        }

    async def _execute_mcp_agent_rollout(
        self,
        sample_id: str,
        user_query: str,
        system_prompt_content: Optional[str],
        openai_formatted_tools: Optional[List[Dict[str, Any]]],
        http_session: aiohttp.ClientSession,
        discovered_tools_for_llm_prompt: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute MCP agent rollout with tool calling."""
        mcp_backend_ref = self.cfg.agent.get("mcp_backend_ref")
        rk_session_id = None
        all_executed_tool_calls_for_sample = []
        final_llm_text_response = None
        final_filesystem_state_from_mcp = None

        # Initial messages for the rollout
        current_messages_for_rollout = []
        if system_prompt_content:
            current_messages_for_rollout.append({"role": "system", "content": system_prompt_content})
        current_messages_for_rollout.append({"role": "user", "content": user_query})

        try:
            backend_requests = [{"backend_name_ref": mcp_backend_ref, "num_instances": 1}]
            assert self.mcp_intermediary_client is not None
            init_response = await self.mcp_intermediary_client.initialize_session(backend_requests)
            if init_response.get("error"):
                raise RuntimeError(
                    f"Main MCP session init failed: {init_response.get('error_details', init_response['error'])}"
                )
            rk_session_id = init_response.get("rk_session_id")

            primary_instance_id_for_agent_actions = None
            initialized_backends = init_response.get("initialized_backends", [])
            if not rk_session_id or not initialized_backends:
                raise RuntimeError(f"Malformed main MCP init response: {init_response}")
            for be_info in initialized_backends:
                if be_info.get("backend_name_ref") == mcp_backend_ref and be_info.get("instances"):
                    primary_instance_id_for_agent_actions = be_info["instances"][0].get("instance_id")
                    break
            if not primary_instance_id_for_agent_actions:
                raise RuntimeError(f"Primary instance ID for agent actions not found for {mcp_backend_ref}")

            logger.info(
                f"Sample {sample_id}: Main MCP session for agent execution. rk_session_id='{rk_session_id}', instance='{primary_instance_id_for_agent_actions}'."
            )

            max_rollout_turns = self.cfg.agent.get("max_rollout_turns", 5)
            final_assistant_output_for_log = None

            for turn_num in range(max_rollout_turns):
                logger.info(
                    f"Sample {sample_id}: Agent Rollout Turn {turn_num + 1}/{max_rollout_turns}. History size: {len(current_messages_for_rollout)}"
                )

                # model_client is initialized when generation is enabled; assert for type-checker
                assert self.model_client is not None
                generation_output_turn = await self.model_client.generate(
                    messages=current_messages_for_rollout,
                    session=http_session,
                    tools=openai_formatted_tools,
                )

                assistant_msg_for_history: Dict[str, Any] = {"role": "assistant"}

                if generation_output_turn.tool_calls:
                    assistant_msg_for_history["tool_calls"] = [
                        tc.model_dump() for tc in generation_output_turn.tool_calls
                    ]
                    current_messages_for_rollout.append(assistant_msg_for_history)
                    final_assistant_output_for_log = json.dumps(assistant_msg_for_history["tool_calls"])

                    for tool_call in generation_output_turn.tool_calls:
                        tool_name = tool_call.function.name
                        tool_call_id = tool_call.id
                        tool_args_dict = None
                        tool_result_content_str = ""
                        try:
                            tool_args_dict = json.loads(tool_call.function.arguments)
                            if not isinstance(tool_args_dict, dict):
                                raise ValueError("Args not dict")

                            assert self.mcp_intermediary_client is not None
                            exec_result = await self.mcp_intermediary_client.call_backend_tool(
                                rk_session_id=rk_session_id,
                                instance_id=primary_instance_id_for_agent_actions,
                                backend_name_ref=mcp_backend_ref,
                                tool_name=tool_name,
                                tool_args=tool_args_dict,
                            )
                            tool_result_content_str = json.dumps(exec_result)
                            all_executed_tool_calls_for_sample.append(
                                {
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "arguments": tool_args_dict,
                                    "result": exec_result,
                                }
                            )
                        except Exception as e_tool_exec:
                            logger.error(
                                f"Sample {sample_id}, Turn {turn_num + 1}: Error executing/parsing tool '{tool_name}': {e_tool_exec}",
                                exc_info=True,
                            )
                            error_payload = {"error": str(e_tool_exec)}
                            if isinstance(e_tool_exec, json.JSONDecodeError):
                                error_payload["detail"] = "Failed to parse arguments string from LLM."
                            tool_result_content_str = json.dumps(error_payload)
                            all_executed_tool_calls_for_sample.append(
                                {
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "arguments_str": tool_call.function.arguments,
                                    "error": str(e_tool_exec),
                                }
                            )

                        current_messages_for_rollout.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_result_content_str,
                            }
                        )

                    if turn_num == max_rollout_turns - 1:
                        logger.warning(f"Sample {sample_id}: Max rollout turns reached after tool call(s).")

                elif generation_output_turn.content:
                    final_llm_text_response = generation_output_turn.content
                    assistant_msg_for_history["content"] = final_llm_text_response
                    current_messages_for_rollout.append(assistant_msg_for_history)
                    final_assistant_output_for_log = final_llm_text_response
                    logger.info(f"Sample {sample_id}, Turn {turn_num + 1}: LLM responded with text. Ending rollout.")
                    break
                else:
                    logger.warning(
                        f"Sample {sample_id}, Turn {turn_num + 1}: LLM provided no content or tool calls. Ending rollout."
                    )
                    final_llm_text_response = "LLM provided no actionable response in this turn."
                    assistant_msg_for_history["content"] = final_llm_text_response
                    current_messages_for_rollout.append(assistant_msg_for_history)
                    final_assistant_output_for_log = final_llm_text_response
                    break

            if (
                not final_llm_text_response
                and not all_executed_tool_calls_for_sample
                and not final_assistant_output_for_log
            ):
                final_assistant_output_for_log = "Agent did not produce text or tool calls within max turns."

            # State Capture
            state_capture_tool = self.cfg.agent.get("state_capture_tool")
            if state_capture_tool:
                state_capture_args = dict(self.cfg.agent.get("state_capture_args", OmegaConf.create({})))
                assert self.mcp_intermediary_client is not None
                final_filesystem_state_from_mcp = await self.mcp_intermediary_client.call_backend_tool(
                    rk_session_id=rk_session_id,
                    instance_id=primary_instance_id_for_agent_actions,
                    backend_name_ref=mcp_backend_ref,
                    tool_name=state_capture_tool,
                    tool_args=state_capture_args,
                )

            return {
                "success": True,
                "final_assistant_output": final_assistant_output_for_log,
                "conversation_history": current_messages_for_rollout,
                "executed_tool_calls": all_executed_tool_calls_for_sample,
                "final_filesystem_state": final_filesystem_state_from_mcp,
            }

        except Exception as e_mcp_main:
            logger.error(
                f"Error during MCP agent main processing for sample {sample_id}: {e_mcp_main}",
                exc_info=True,
            )
            return {
                "success": False,
                "error": f"MCP agent processing failed: {str(e_mcp_main)}",
            }
        finally:
            if rk_session_id and self.mcp_intermediary_client:
                assert self.mcp_intermediary_client is not None
                await self.mcp_intermediary_client.cleanup_session(rk_session_id)

    async def _process_single_sample(
        self,
        sample: Dict[str, Any],
        http_session: Optional[aiohttp.ClientSession],  # For model_client, not mcp_client
        original_index: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Main entry point for processing a single sample, handling N-variant generation."""
        sample_id_fallback = (
            f"idx_{original_index}" if original_index is not None else "unknown_id_" + os.urandom(4).hex()
        )
        sample_id = sample.get("id", sample_id_fallback)
        user_query = sample.get("user_query")
        ground_truth_for_eval = sample.get("ground_truth_for_eval")
        existing_messages = sample.get("messages")

        # Check for N-variant generation
        n_variants = self.cfg.generation.get("n", 1)
        if not isinstance(n_variants, int) or n_variants < 1:
            n_variants = 1

        # If N > 1, generate multiple variants
        if n_variants > 1:
            return await self._process_n_variants(
                sample=sample,
                sample_id=sample_id,
                user_query=user_query,
                ground_truth_for_eval=ground_truth_for_eval,
                existing_messages=existing_messages,
                http_session=http_session,
                n_variants=n_variants,
                original_index=original_index,
            )

        # Single variant processing
        return await self._process_single_sample_internal(
            sample=sample,
            http_session=http_session,
            original_index=original_index,
        )

    async def _process_single_sample_internal(
        self,
        sample: Dict[str, Any],
        http_session: Optional[aiohttp.ClientSession],  # For model_client, not mcp_client
        original_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        sample_id_fallback = (
            f"idx_{original_index}" if original_index is not None else "unknown_id_" + os.urandom(4).hex()
        )
        sample_id = sample.get("id", sample_id_fallback)
        user_query = sample.get("user_query")
        ground_truth_for_eval = sample.get("ground_truth_for_eval")
        existing_messages = sample.get("messages")

        # Check if we have either the generation format (user_query + ground_truth)
        # or the evaluation format (existing messages)
        has_generation_format = user_query is not None and ground_truth_for_eval is not None
        has_evaluation_format = existing_messages is not None

        if not has_generation_format and not has_evaluation_format:
            logger.warning(
                f"Skipping sample {sample_id}: needs either ('user_query' + 'ground_truth_for_eval') for generation or 'messages' for evaluation."
            )
            return {
                "id": sample_id,
                "error": "Missing required fields for generation/evaluation",
            }

        original_system_prompt = sample.get("system_prompt") or self.cfg.get("system_prompt")
        discovered_tools_for_llm_prompt: List[Dict[str, Any]] = []
        openai_formatted_tools: Optional[List[Dict[str, Any]]] = None

        # This variable will hold the final assistant response string for top-level logging/preview
        # It might be a text response or a JSON string of the last tool call request by LLM.
        final_assistant_output_for_log: Optional[str] = None

        # --- Pre-generation: Tool Discovery (if MCP agent) ---
        if self.mcp_intermediary_client and self.cfg.agent.type == "mcp_agent":
            mcp_backend_ref_for_tools = self.cfg.agent.get("mcp_backend_ref")
            if not mcp_backend_ref_for_tools:
                raise ValueError("agent.mcp_backend_ref must be configured for mcp_agent tool discovery.")
            discovered_tools_for_llm_prompt = await self._discover_tools_for_sample(
                sample_id, mcp_backend_ref_for_tools
            )

        # --- Construct System Prompt and Format Tools for LLM ---
        system_prompt_content = original_system_prompt
        if self.mcp_intermediary_client and self.cfg.agent.type == "mcp_agent" and discovered_tools_for_llm_prompt:
            openai_formatted_tools = []
            for mcp_tool_dict in discovered_tools_for_llm_prompt:
                input_schema = mcp_tool_dict.get("inputSchema", {})
                openai_formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mcp_tool_dict.get("name", "unknown"),
                            "description": mcp_tool_dict.get("description", ""),
                            "parameters": input_schema,
                        },
                    }
                )
            if original_system_prompt:
                system_prompt_content = (
                    f"{original_system_prompt}\n\nYou have access to tools. Use them if appropriate."
                )
            else:
                system_prompt_content = "You are a helpful assistant with access to tools. Use them if appropriate."

        # Handle existing messages for evaluation vs building new messages for generation
        if has_evaluation_format and not has_generation_format:
            # Evaluation mode: use existing messages, skip generation
            if not self.cfg.generation.enabled:
                # Pass raw messages as-is for evaluation mode - the @reward_function decorator
                # will handle conversion based on the function's type annotations
                final_messages_for_eval = existing_messages

                # Call reward function directly with existing messages
                eval_params = dict(self.cfg.reward.get("params", OmegaConf.create({})))

                # Extract per-sample values and add them to eval_params
                sample_agent_id = sample.get("agent_id")
                sample_test_id = sample.get("test_id")
                if sample_agent_id is not None:
                    eval_params["agent_id"] = sample_agent_id
                if sample_test_id is not None:
                    eval_params["test_id"] = sample_test_id

                eval_result_obj = self.reward_function(
                    messages=final_messages_for_eval,
                    ground_truth=ground_truth_for_eval,
                    final_filesystem_state=None,
                    **eval_params,
                )

                return {
                    "id": sample_id,
                    "user_query": None,  # Not applicable for evaluation mode
                    "system_prompt": system_prompt_content,
                    "assistant_response": None,  # Not applicable for evaluation mode
                    "ground_truth_for_eval": ground_truth_for_eval,
                    "evaluation_score": eval_result_obj.score,
                    "evaluation_reason": eval_result_obj.reason,
                    "evaluation_metrics": (
                        {k: v.model_dump() for k, v in eval_result_obj.metrics.items()}
                        if eval_result_obj.metrics
                        else {}
                    ),
                    "full_conversation_history": existing_messages,  # Store original messages
                }
            else:
                logger.warning(f"Sample {sample_id}: Evaluation mode requires generation.enabled=false")
                return {
                    "id": sample_id,
                    "error": "Evaluation mode requires generation.enabled=false",
                }

        # Generation mode: Initial messages for the main rollout (or single generation if not agent)
        # At this point, generation format is guaranteed by the control flow above; cast for type checking
        user_query_str: str = cast(str, user_query)
        current_messages_for_rollout: List[Dict[str, Any]] = []
        if system_prompt_content:
            current_messages_for_rollout.append({"role": "system", "content": system_prompt_content})
        current_messages_for_rollout.append({"role": "user", "content": user_query_str})

        # --- LLM Generation / Agent Rollout ---
        if not self.cfg.generation.enabled:
            # ... (existing logic for disabled generation, using assistant_response_content from sample or cache) ...
            # This part needs to ensure final_assistant_output_for_log is set.
            # For brevity, assuming this part correctly sets final_assistant_output_for_log if generation is disabled.
            assistant_response_col_name = self.cfg.dataset.get("column_mapping", {}).get(
                "assistant_response_column", "assistant_response"
            )
            final_assistant_output_for_log = sample.get(assistant_response_col_name)
            # ... (rest of the non-generation logic)
            if not final_assistant_output_for_log:  # Try cache if generation disabled and no direct column
                gen_cfg = self.cfg.generation
                final_assistant_output_for_log = self.cache.get(
                    sample_id=sample_id,
                    system_prompt=original_system_prompt,
                    user_query=cast(str, user_query),
                    model_name=gen_cfg.get("model_name", "unknown_model"),
                    temperature=gen_cfg.get("temperature", 0.0),
                    top_p=gen_cfg.get("top_p", 1.0),
                    top_k=gen_cfg.get("top_k", 0),
                    min_p=gen_cfg.get("min_p", 0.0),
                    max_tokens=gen_cfg.get("max_tokens", 2048),
                    reasoning_effort=gen_cfg.get("reasoning_effort", None),
                )
            if not final_assistant_output_for_log:
                return {
                    "id": sample_id,
                    "error": "No response (gen disabled, not in sample/cache)",
                    "evaluation_score": 0.0,
                }

        elif not self.model_client or not http_session:  # Generation enabled but client/session missing
            return {
                "id": sample_id,
                "error": "Generation client/session not configured",
            }

        # --- MCP Agent Rollout Loop ---
        elif self.mcp_intermediary_client and self.cfg.agent.type == "mcp_agent":
            mcp_result = await self._execute_mcp_agent_rollout(
                sample_id=sample_id,
                user_query=user_query_str,
                system_prompt_content=system_prompt_content,
                openai_formatted_tools=openai_formatted_tools,
                http_session=http_session,
                discovered_tools_for_llm_prompt=discovered_tools_for_llm_prompt,
            )

            if not mcp_result["success"]:
                return {
                    "id": sample_id,
                    "error": mcp_result["error"],
                    "discovered_tools": discovered_tools_for_llm_prompt,
                }

            final_assistant_output_for_log = mcp_result["final_assistant_output"]
            current_messages_for_rollout = mcp_result["conversation_history"]
            all_executed_tool_calls_for_sample = mcp_result["executed_tool_calls"]
            final_filesystem_state_from_mcp = mcp_result["final_filesystem_state"]

            # Evaluation based on the final state and conversation
            eval_params = dict(self.cfg.reward.get("params", OmegaConf.create({})))
            eval_result_obj = self.reward_function(
                messages=[Message(**msg) for msg in current_messages_for_rollout],
                ground_truth=ground_truth_for_eval,
                final_filesystem_state=final_filesystem_state_from_mcp,
                **eval_params,
            )

            return {
                "id": sample_id,
                "user_query": user_query,
                "system_prompt": system_prompt_content,
                "assistant_response": final_assistant_output_for_log,
                "full_conversation_history": current_messages_for_rollout,
                "ground_truth_for_eval": ground_truth_for_eval,
                "discovered_tools": discovered_tools_for_llm_prompt,
                "executed_tool_calls": all_executed_tool_calls_for_sample,
                "final_mcp_state_captured": final_filesystem_state_from_mcp or "Not captured",
                "evaluation_score": eval_result_obj.score,
                "evaluation_reason": eval_result_obj.reason,
                "evaluation_metrics": (
                    {k: v.model_dump() for k, v in eval_result_obj.metrics.items()} if eval_result_obj.metrics else {}
                ),
            }

        # --- Standard LLM Generation (Non-Agent) ---
        else:
            generation_result = await self._execute_standard_generation(
                sample_id=sample_id,
                user_query=user_query_str,
                system_prompt_content=system_prompt_content,
                http_session=http_session,
            )

            if not generation_result["success"]:
                return {
                    "id": sample_id,
                    "error": "Standard generation failed",
                    "evaluation_score": 0.0,
                }

            final_assistant_output_for_log = generation_result["final_assistant_output"]
            conversation_history = generation_result["conversation_history"]

            # Construct final_messages_for_eval for standard evaluation
            final_messages_for_eval = [Message(**msg) for msg in conversation_history]

            eval_params = dict(self.cfg.reward.get("params", OmegaConf.create({})))
            eval_result_obj = self.reward_function(
                messages=final_messages_for_eval,
                ground_truth=ground_truth_for_eval,
                final_filesystem_state=None,
                **eval_params,
            )
            return {
                "id": sample_id,
                "user_query": user_query,
                "system_prompt": system_prompt_content,
                "assistant_response": final_assistant_output_for_log,
                "ground_truth_for_eval": ground_truth_for_eval,
                "evaluation_score": eval_result_obj.score,
                "evaluation_reason": eval_result_obj.reason,
                "evaluation_metrics": (
                    {k: v.model_dump() for k, v in eval_result_obj.metrics.items()} if eval_result_obj.metrics else {}
                ),
            }

        # Fallback if logic didn't hit a return, though it should.
        return {
            "id": sample_id,
            "error": "Processing logic incomplete",
            "evaluation_score": 0.0,
        }

    async def run(self) -> List[Dict[str, Any]]:
        logger.info("Starting evaluation pipeline run...")

        try:
            prompt_dataset_config = self.cfg.dataset
            prompt_dataset = hydra.utils.instantiate(prompt_dataset_config)

            if isinstance(prompt_dataset, DatasetDict):
                split_name = prompt_dataset_config.get("split", "train")
                if split_name in prompt_dataset:
                    prompt_dataset = prompt_dataset[split_name]
                else:
                    logger.error(f"Split '{split_name}' not found. Available: {list(prompt_dataset.keys())}")
                    return []
            elif not isinstance(prompt_dataset, Dataset):
                logger.error(f"Loaded dataset is not a Hugging Face Dataset. Type: {type(prompt_dataset)}")
                return []

            dataset_source = getattr(
                prompt_dataset_config,
                "path_or_name",
                getattr(prompt_dataset_config, "base_dataset", "dataset"),
            )
            logger.info(f"Loaded {len(prompt_dataset)} samples from {dataset_source}.")
        except InstantiationException as ie:
            final_cause = ie
            while final_cause.__cause__ is not None:
                final_cause = final_cause.__cause__
            if (
                isinstance(final_cause, ValueError)
                and str(final_cause) == "Invalid pattern: '**' can only be an entire path component"
            ):
                base_dataset_config_name = prompt_dataset_config.get("base_dataset", "UnknownBaseDatasetConfig")
                dataset_display_name = base_dataset_config_name
                helpful_message = (
                    f"Failed to load the base dataset specified as '{dataset_display_name}' in your derived dataset configuration. "
                    f"This occurred due to an internal error in the 'datasets' library (via fsspec): '{str(final_cause)}'.\n"
                    "The error message \"Invalid pattern: '**' can only be an entire path component\" often indicates issues with "
                    "how the 'datasets' library is resolving the path to the dataset, potential Hugging Face Hub connectivity/authentication problems, or a corrupted local cache.\n\n"
                    "Please try the following troubleshooting steps:\n"
                    "1. Verify Hugging Face Hub Token: Ensure your token is correctly configured (e.g., run `huggingface-cli login`).\n"
                    "2. Clear Datasets Cache: Try removing the subdirectory related to the actual Hugging Face dataset path/name from `~/.cache/huggingface/datasets/`.\n"
                    "3. Update Libraries: `pip install --upgrade datasets huggingface_hub fsspec`.\n"
                    "4. Test Direct Loading: (See previous detailed instructions for direct loading test script).\n"
                    f"Original InstantiationException details: {ie}"
                )
                logger.error(helpful_message, exc_info=False)
            else:
                logger.error(f"Failed to load prompt dataset: {ie}", exc_info=True)
            return []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during dataset loading: {e}",
                exc_info=True,
            )
            return []

        all_results: List[Dict[str, Any]] = []
        limit_samples = self.cfg.evaluation_params.get("limit_samples", None)
        samples_to_process_count = len(prompt_dataset)
        if limit_samples is not None and limit_samples > 0:
            samples_to_process_count = min(len(prompt_dataset), limit_samples)

        logger.info(f"Processing {samples_to_process_count} samples.")

        http_session_for_model_client: Optional[aiohttp.ClientSession] = None  # Renamed for clarity
        if self.cfg.generation.enabled and self.model_client:
            http_session_for_model_client = aiohttp.ClientSession()

        if self.cfg.get("agent") and self.cfg.agent.get("type") == "mcp_agent":
            self.mcp_intermediary_client = IntermediaryMCPClient(
                intermediary_server_url=self.cfg.agent.intermediary_server_url
            )
            logger.info(f"Created IntermediaryMCPClient instance with URL: {self.cfg.agent.intermediary_server_url}")

        async def execute_tasks():
            tasks = []
            # http_session_for_model_client is managed outside this async def now

            max_concurrent = self.cfg.generation.api_params.get("max_concurrent_requests", 5)
            if not isinstance(max_concurrent, int) or max_concurrent <= 0:
                logger.warning(f"Invalid max_concurrent_requests value ({max_concurrent}), defaulting to 5.")
                max_concurrent = 5
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore_wrapper(sample_idx: int, sample_data: Dict[str, Any]):
                prelim_sample_id = sample_data.get("id", f"idx_{sample_idx}")
                async with semaphore:
                    logger.info(f"Concurrency slot acquired for sample '{prelim_sample_id}', attempting to process.")
                    return await self._process_single_sample(
                        sample_data,
                        http_session_for_model_client,
                        original_index=sample_idx,
                    )

            for i in range(samples_to_process_count):
                tasks.append(process_with_semaphore_wrapper(i, prompt_dataset[i]))

            batch_size_for_logging = self.cfg.logging_params.get("batch_log_interval", 10)
            if not isinstance(batch_size_for_logging, int) or batch_size_for_logging <= 0:
                logger.warning(f"Invalid batch_log_interval ({batch_size_for_logging}), defaulting to 10.")
                batch_size_for_logging = 10

            for i_outer in range(0, len(tasks), batch_size_for_logging):
                batch_tasks = tasks[i_outer : i_outer + batch_size_for_logging]
                # asyncio.gather with return_exceptions=True returns List[Any]; cast to expected union
                batch_results_values = cast(
                    List[Union[Exception, Dict[str, Any], List[Dict[str, Any]]]],
                    await asyncio.gather(*batch_tasks, return_exceptions=True),
                )
                for res_idx, res_or_exc in enumerate(batch_results_values):
                    if isinstance(res_or_exc, Exception):
                        logger.error(
                            f"Task for sample index {i_outer + res_idx} failed: {res_or_exc}",
                            exc_info=True,
                        )
                        all_results.append(
                            {
                                "id": prompt_dataset[i_outer + res_idx].get("id", "unknown"),
                                "error": str(res_or_exc),
                            }
                        )
                    elif res_or_exc is not None:
                        # Handle both single results and lists of results (N-variant)
                        if isinstance(res_or_exc, list):
                            all_results.extend(res_or_exc)
                        else:
                            # res_or_exc is a Dict[str, Any] here
                            all_results.append(res_or_exc)  # type: ignore[arg-type]
                logger.info(
                    f"Completed batch up to sample {i_outer + len(batch_tasks)}. Total results/errors: {len(all_results)}"
                )

        try:
            if self.mcp_intermediary_client:
                async with self.mcp_intermediary_client:
                    await execute_tasks()
            else:
                await execute_tasks()
        finally:
            if http_session_for_model_client:
                await http_session_for_model_client.close()
                logger.debug("Closed aiohttp.ClientSession for model_client in main run() finally block.")

        output_file_path = self.cfg.output.get("results_file", None)
        if output_file_path:
            if not os.path.isabs(output_file_path) and self.cfg.hydra_output_dir:
                output_file_path = os.path.join(self.cfg.hydra_output_dir, output_file_path)
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    for result_item in all_results:
                        f.write(json.dumps(result_item) + "\n")
                logger.info(f"Detailed results saved to: {os.path.abspath(output_file_path)}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file_path}: {e}")

        preview_pairs_file_path = self.cfg.output.get("preview_pairs_file", "preview_input_output_pairs.jsonl")
        if preview_pairs_file_path:
            if not os.path.isabs(preview_pairs_file_path) and self.cfg.hydra_output_dir:
                preview_pairs_file_path = os.path.join(self.cfg.hydra_output_dir, preview_pairs_file_path)
            preview_data_to_save = []
            for result_item in all_results:
                # Use full_conversation_history if available, otherwise construct from system/user/assistant
                if "full_conversation_history" in result_item:
                    messages_to_save = result_item["full_conversation_history"]
                elif (
                    "error" in result_item
                    or not result_item.get("user_query")
                    or not result_item.get("assistant_response")
                ):
                    continue  # Skip if essential parts for basic preview are missing and no history
                else:  # Construct basic messages for non-agent or simple agent cases
                    messages_to_save = []
                    if result_item.get("system_prompt"):
                        messages_to_save.append({"role": "system", "content": result_item["system_prompt"]})
                    messages_to_save.append({"role": "user", "content": result_item["user_query"]})
                    messages_to_save.append(
                        {
                            "role": "assistant",
                            "content": result_item["assistant_response"],
                        }
                    )

                pair_item = {"messages": messages_to_save}
                if result_item.get("ground_truth_for_eval"):
                    pair_item["ground_truth"] = result_item["ground_truth_for_eval"]
                if result_item.get("id"):
                    pair_item["id"] = result_item["id"]
                preview_data_to_save.append(pair_item)

            if preview_data_to_save:
                try:
                    os.makedirs(os.path.dirname(preview_pairs_file_path), exist_ok=True)
                    with open(preview_pairs_file_path, "w", encoding="utf-8") as f:
                        for item in preview_data_to_save:
                            f.write(json.dumps(item) + "\n")
                    logger.info(f"Input/output pairs for preview saved to: {os.path.abspath(preview_pairs_file_path)}")
                except Exception as e:
                    logger.error(f"Failed to save preview pairs to {preview_pairs_file_path}: {e}")

        logger.info("Evaluation pipeline run finished.")
        return all_results
