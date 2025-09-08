import asyncio
import time
from typing import List

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - optional dependency path
    # Minimal fallback base type to satisfy typing when langchain is not present
    class BaseMessage:  # type: ignore
        pass


from eval_protocol.models import EvaluationRow, Message
from openai.types import CompletionUsage
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


class LangGraphRolloutProcessor(RolloutProcessor):
    """Generic rollout processor for LangChain agents.

    Accepts an async factory that returns a target to invoke. The target can be:
    - An object with `.graph.ainvoke(payload)` (e.g., LangGraph compiled graph)
    - An object with `.ainvoke(payload)`
    - A callable that accepts `payload` and returns the result dict
    """

    def __init__(self, get_invoke_target):
        self.get_invoke_target = get_invoke_target

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig):
        tasks: List[asyncio.Task] = []

        async def _process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            # Build LC messages from EP row
            try:
                from langchain_core.messages import HumanMessage
            except Exception:
                # Fallback minimal message if langchain_core is unavailable
                class HumanMessage(BaseMessage):  # type: ignore
                    def __init__(self, content: str):
                        self.content = content
                        self.type = "human"

            lm_messages: List[BaseMessage] = []
            if row.messages:
                last_user = [m for m in row.messages if m.role == "user"]
                if last_user:
                    content = last_user[-1].content or ""
                    if isinstance(content, list):
                        # Flatten our SDK content parts into a single string for LangChain
                        content = "".join([getattr(p, "text", str(p)) for p in content])
                    lm_messages.append(HumanMessage(content=str(content)))
            if not lm_messages:
                lm_messages = [HumanMessage(content="")]  # minimal

            target = await self.get_invoke_target(config)

            # Resolve the appropriate async invoke function
            if hasattr(target, "graph") and hasattr(target.graph, "ainvoke"):

                async def _invoke_graph(payload):
                    return await target.graph.ainvoke(payload)  # type: ignore[attr-defined]

                invoke_fn = _invoke_graph
            elif hasattr(target, "ainvoke"):

                async def _invoke_direct(payload):
                    return await target.ainvoke(payload)  # type: ignore[attr-defined]

                invoke_fn = _invoke_direct
            elif callable(target):
                # If target is a normal callable, call it directly; if it returns an awaitable, await it
                async def _invoke_wrapper(payload):
                    result = target(payload)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result

                invoke_fn = _invoke_wrapper
            else:
                raise TypeError("Unsupported invoke target for LangGraphRolloutProcessor")

            result_obj = await invoke_fn({"messages": lm_messages})
            # Accept both dicts and objects with .get/.messages
            if isinstance(result_obj, dict):
                result_messages: List[BaseMessage] = result_obj.get("messages", [])
            else:
                result_messages = getattr(result_obj, "messages", [])

            # TODO: i didn't see a langgraph example so couldn't fully test this. should uncomment and test when we have example ready.
            # total_input_tokens = 0
            # total_output_tokens = 0
            # total_tokens = 0

            # for msg in result_messages:
            #     if isinstance(msg, BaseMessage):
            #         usage = getattr(msg, 'response_metadata', {})
            #     else:
            #         usage = msg.get("response_metadata", {})

            #     if usage:
            #         total_input_tokens += usage.get("prompt_tokens", 0)
            #         total_output_tokens += usage.get("completion_tokens", 0)
            #         total_tokens += usage.get("total_tokens", 0)

            # row.execution_metadata.usage = CompletionUsage(
            #     prompt_tokens=total_input_tokens,
            #     completion_tokens=total_output_tokens,
            #     total_tokens=total_tokens,
            # )

            def _serialize_message(msg: BaseMessage) -> Message:
                # Prefer SDK-level serializer
                try:
                    from eval_protocol.adapters.langchain import serialize_lc_message_to_ep as _ser

                    return _ser(msg)
                except Exception:
                    # Minimal fallback: best-effort string content only
                    content = getattr(msg, "content", "")
                    return Message(role=getattr(msg, "type", "assistant"), content=str(content))

            row.messages = [_serialize_message(m) for m in result_messages]

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            return row

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        return None
