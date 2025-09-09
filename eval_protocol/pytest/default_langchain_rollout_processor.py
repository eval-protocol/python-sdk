import asyncio
import time
from typing import List, Any, cast

try:
    from langchain_core.messages import BaseMessage as LCBaseMessage, HumanMessage  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    # Minimal fallbacks to satisfy typing when langchain is not present
    class LCBaseMessage:  # type: ignore
        content: str
        type: str

        def __init__(self, content: str = "", msg_type: str = "assistant"):
            self.content = content
            self.type = msg_type

    class HumanMessage(LCBaseMessage):  # type: ignore
        def __init__(self, content: str):
            super().__init__(content=content, msg_type="human")


from eval_protocol.models import EvaluationRow, Message
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

            # Build LC messages from EP row (minimal: last user to HumanMessage)
            lm_messages: List[LCBaseMessage] = []
            if row.messages:
                last_user = [m for m in row.messages if m.role == "user"]
                if last_user:
                    content = last_user[-1].content or ""
                    if isinstance(content, list):
                        content = "".join([getattr(p, "text", str(p)) for p in content])
                    lm_messages.append(HumanMessage(content=str(content)))
            if not lm_messages:
                lm_messages = [HumanMessage(content="")]

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

                async def _invoke_wrapper(payload):
                    result = target(payload)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result

                invoke_fn = _invoke_wrapper
            else:
                raise TypeError("Unsupported invoke target for LangGraphRolloutProcessor")

            result_obj = await invoke_fn({"messages": lm_messages})
            if isinstance(result_obj, dict):
                result_messages: List[LCBaseMessage] = result_obj.get("messages", [])
            else:
                result_messages = getattr(result_obj, "messages", [])

            def _serialize_message(msg: LCBaseMessage) -> Message:
                try:
                    from eval_protocol.adapters.langchain import serialize_lc_message_to_ep as _ser
                except ImportError:
                    content = getattr(msg, "content", "")
                    return Message(role=getattr(msg, "type", "assistant"), content=str(content))
                return _ser(cast(Any, msg))

            row.messages = [_serialize_message(m) for m in result_messages]

            row.execution_metadata.duration_seconds = time.perf_counter() - start_time

            return row

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        return None
