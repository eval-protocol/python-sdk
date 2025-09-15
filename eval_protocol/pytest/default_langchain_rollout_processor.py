import asyncio
from typing import Any, Callable, Dict, List, Optional

from eval_protocol.models import EvaluationRow, Status, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import CompletionParams, RolloutProcessorConfig


class LangGraphRolloutProcessor(RolloutProcessor):
    """
    Generic rollout processor for LangGraph graphs.

    Configure with:
    - to_input(row): build the input payload for graph.ainvoke (default: {"messages": row.messages})
    - apply_result(row, result): write graph outputs back onto the row (default: row.messages = result["messages"])
    - build_graph_kwargs(cp): map completion_params to graph kwargs (default: {})

    Compatible with eval_protocol.pytest.evaluation_test.
    """

    def __init__(
        self,
        *,
        # Factory must accept RolloutProcessorConfig (parity with Pydantic AI processor)
        graph_factory: Callable[[RolloutProcessorConfig], Any],
        to_input: Optional[Callable[[EvaluationRow], Dict[str, Any]]] = None,
        apply_result: Optional[Callable[[EvaluationRow, Any], EvaluationRow]] = None,
        build_graph_kwargs: Optional[Callable[[CompletionParams], Dict[str, Any]]] = None,
        input_key: str = "messages",
        output_key: str = "messages",
        # Optional: build per-invoke RunnableConfig dict from full RolloutProcessorConfig
        build_invoke_config: Optional[Callable[[RolloutProcessorConfig], Dict[str, Any]]] = None,
    ) -> None:
        # Build the graph per-call using completion_params
        self._graph_factory = graph_factory
        self._to_input = to_input
        self._apply_result = apply_result
        self._build_graph_kwargs = build_graph_kwargs
        self._input_key = input_key
        self._output_key = output_key
        self._build_invoke_config = build_invoke_config

    def _default_to_input(self, row: EvaluationRow) -> Dict[str, Any]:
        messages = row.messages or []
        from eval_protocol.adapters.langchain import serialize_ep_messages_to_lc as _to_lc

        return {self._input_key: _to_lc(messages)}

    def _default_apply_result(self, row: EvaluationRow, result: Any) -> EvaluationRow:
        # Expect dict with output_key → list of messages; coerce to EP messages
        maybe_msgs = None
        if isinstance(result, dict):
            maybe_msgs = result.get(self._output_key)

        if maybe_msgs is None:
            return row

        # If already EP messages, assign directly
        if isinstance(maybe_msgs, list) and all(isinstance(m, Message) for m in maybe_msgs):
            row.messages = maybe_msgs
            return row

        # Try to convert from LangChain messages; preserve EP Message items as-is
        try:
            from langchain_core.messages import BaseMessage as _LCBase
            from eval_protocol.adapters.langchain import serialize_lc_message_to_ep as _to_ep

            if isinstance(maybe_msgs, list) and any(isinstance(m, _LCBase) for m in maybe_msgs):
                converted: List[Message] = []
                for m in maybe_msgs:
                    if isinstance(m, Message):
                        converted.append(m)
                    elif isinstance(m, _LCBase):
                        converted.append(_to_ep(m))
                    elif isinstance(m, dict):
                        role = m.get("role") or "assistant"
                        content = m.get("content")
                        tool_calls = m.get("tool_calls")
                        function_call = m.get("function_call")
                        converted.append(
                            Message(role=role, content=content, tool_calls=tool_calls, function_call=function_call)
                        )
                    else:
                        # Best-effort for LC-like objects without importing LC types
                        role_like = getattr(m, "type", None)
                        content_like = getattr(m, "content", None)
                        if content_like is not None:
                            role_value = "assistant"
                            if isinstance(role_like, str):
                                rl = role_like.lower()
                                if rl in ("human", "user"):
                                    role_value = "user"
                                elif rl in ("ai", "assistant"):
                                    role_value = "assistant"
                                elif rl in ("system",):
                                    role_value = "system"
                            converted.append(Message(role=role_value, content=str(content_like)))
                        else:
                            converted.append(Message(role="assistant", content=str(m)))
                row.messages = converted
                return row
        except ImportError:
            # If LC is not available, fall back to best-effort below
            pass

        # Generic best-effort fallback: stringify to assistant messages
        if isinstance(maybe_msgs, list):
            row.messages = [Message(role="assistant", content=str(m)) for m in maybe_msgs]
        else:
            row.messages = [Message(role="assistant", content=str(maybe_msgs))]
        return row

    def _default_build_graph_kwargs(self, _: CompletionParams) -> Dict[str, Any]:
        # Keep generic: callers can override to map to their graph’s expected kwargs
        return {}

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        tasks: List[asyncio.Task[EvaluationRow]] = []

        to_input = self._to_input or self._default_to_input
        apply_result = self._apply_result or self._default_apply_result
        build_kwargs = self._build_graph_kwargs or self._default_build_graph_kwargs

        graph_config: Optional[Dict[str, Any]] = None
        if config.completion_params:
            graph_config = build_kwargs(config.completion_params)

        # (Re)build the graph for this call using the full typed config.
        graph_target = self._graph_factory(config)

        # Build per-invoke config if provided; otherwise reuse graph_config for backwards compat
        invoke_config: Optional[Dict[str, Any]] = None
        if self._build_invoke_config is not None:
            invoke_config = self._build_invoke_config(config)
        elif graph_config is not None:
            invoke_config = graph_config

        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            try:
                payload = to_input(row)
                if invoke_config is not None:
                    result = await graph_target.ainvoke(payload, config=invoke_config)
                else:
                    result = await graph_target.ainvoke(payload)
                row = apply_result(row, result)
                row.rollout_status = Status.rollout_finished()
                return row
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as e:  # noqa: BLE001
                row.rollout_status = Status.rollout_error(str(e))
                return row

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        # No-op by default
        return None
