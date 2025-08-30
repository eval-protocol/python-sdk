import asyncio
from typing import List

from langchain_core.messages import BaseMessage

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
            # Build LC messages from EP row
            from langchain_core.messages import HumanMessage

            lm_messages: List[BaseMessage] = []
            if row.messages:
                last_user = [m for m in row.messages if m.role == "user"]
                if last_user:
                    lm_messages.append(HumanMessage(content=last_user[-1].content or ""))
            if not lm_messages:
                lm_messages = [HumanMessage(content="")]  # minimal

            target = await self.get_invoke_target(config)

            # Resolve the appropriate async invoke function
            if hasattr(target, "graph") and hasattr(target.graph, "ainvoke"):
                invoke_fn = target.graph.ainvoke
            elif hasattr(target, "ainvoke"):
                invoke_fn = target.ainvoke
            elif callable(target):

                async def _invoke_wrapper(payload):
                    return await target(payload)

                invoke_fn = _invoke_wrapper
            else:
                raise TypeError("Unsupported invoke target for LangGraphRolloutProcessor")

            result = await invoke_fn({"messages": lm_messages})
            result_messages: List[BaseMessage] = result.get("messages", [])

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
            return row

        for r in rows:
            tasks.append(asyncio.create_task(_process_row(r)))

        return tasks

    def cleanup(self) -> None:
        return None
