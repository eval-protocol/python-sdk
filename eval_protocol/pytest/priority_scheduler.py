import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union

from tqdm.asyncio import tqdm as async_tqdm

from eval_protocol.models import EvaluationRow, Status
from eval_protocol.pytest.types import RolloutProcessorConfig, TestFunction
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.evaluation_test_utils import rollout_processor_with_retry, add_cost_metrics
from eval_protocol.pytest.buffer import MicroBatchDataBuffer
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.human_id import generate_id
from eval_protocol.log_utils.rollout_context import rollout_logging_context
from eval_protocol.pytest.execution import execute_pytest_with_exception_handling

ENABLE_SPECULATION = os.getenv("ENABLE_SPECULATION", "0").strip() == "1"

@dataclass(order=True)
class RolloutTask:
    """
    Represents a single unit of work for the worker pool.
    Priority tuple structure: (status, row_index)
      - status: 0 = High Priority (e.g., subsequent micro-batches of an already started sample)
                1 = Low Priority (e.g., starting a new sample)
      - row_index: Used to maintain dataset order for initial scheduling
    """
    priority: tuple[int, int]
    
    # Payload (excluded from comparison)
    row: EvaluationRow = field(compare=False)
    run_indices: List[int] = field(compare=False)  # Which runs to execute in this task
    config: RolloutProcessorConfig = field(compare=False)
    row_index: int = field(compare=False) # To track which sample this belongs to
    
    # History for speculation (injected from previous micro-batches)
    history: List[str] = field(compare=False, default_factory=list)

class PriorityRolloutScheduler:
    """
    Manages a priority queue of rollout tasks and a pool of workers.
    Ensures that once a sample starts processing, its subsequent micro-batches
    are prioritized to complete the sample as quickly as possible.
    """
    def __init__(
        self,
        rollout_processor: RolloutProcessor,
        max_concurrent_rollouts: int,
        active_logger: DatasetLogger,
        max_concurrent_evaluations: int,
        eval_executor: TestFunction, # Callback to run evaluation
        output_buffer: Optional[MicroBatchDataBuffer] = None,
        rollout_n: int = 0,
        mode: str = "pointwise",
        in_group_minibatch_size: int = 0, # for one sample, how many runs to execute at the same time
        evaluation_test_kwargs: Dict[str, Any] = {},
    ):
        self.rollout_processor = rollout_processor
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.active_logger = active_logger
        self.eval_executor = eval_executor
        self.output_buffer = output_buffer
        self.mode = mode
        
        # Priority Queue: Stores RolloutTask
        self.queue: asyncio.PriorityQueue[RolloutTask] = asyncio.PriorityQueue()
        
        # Concurrency Control
        self.eval_sem = asyncio.Semaphore(max_concurrent_evaluations)
        
        # Results storage
        self.results: List[EvaluationRow] = [] # for backward compatibility reason, we save all results here to return
        self.groups_buffer: Dict[int, List[EvaluationRow]] = defaultdict(list) # buffer for group results. only flush to output buffer when a whole group is ready
        
        self.background_tasks = set() # run evaluations in the background asynchronously
        
        self.rollout_n = rollout_n
        self.in_group_minibatch_size = in_group_minibatch_size if in_group_minibatch_size > 0 else rollout_n
        self.evaluation_test_kwargs = evaluation_test_kwargs
        
        # Progress bars (initialized in run())
        self.rollout_pbar: Optional[async_tqdm] = None
        self.eval_pbar: Optional[async_tqdm] = None
        
        # Track active rollouts: {row_index: set of run_indices currently in progress}
        self.active_rollouts: Dict[int, set] = defaultdict(set)
        self.active_rollouts_lock = asyncio.Lock()
        
        # Track active evaluations
        self.active_evals: int = 0
        self.active_evals_lock = asyncio.Lock()

    async def schedule_dataset(
        self,
        dataset: List[EvaluationRow],
        base_config: RolloutProcessorConfig,
    ):
        """
        Populates the queue with initial tasks (the first micro-batch for each sample).
        """
        for i, row in enumerate(dataset):
            # Calculate ranges for the first in-group minibatch
            batch_start = 0
            batch_end = min(self.in_group_minibatch_size, self.rollout_n)
            run_indices = list(range(batch_start, batch_end))
            
            # Initial priority: Low (1), ordered by dataset index
            priority = (1, i)
            
            task = RolloutTask(
                priority=priority,
                row=row,
                run_indices=run_indices,
                config=base_config,
                row_index=i,
                history=[] # Initial batch has no history
            )
            self.queue.put_nowait(task)

    async def worker(self):
        """
        Worker loop: fetch task -> execute micro-batch -> schedule next batch (if any).
        """
        while True:
            # Get a task from the priority queue    
            task: RolloutTask = await self.queue.get()

            try:
                await self._process_task(task)
            except Exception as e:
                logging.error(f"Error processing task for row {task.row.input_metadata.row_id}: {e}", exc_info=True)
            finally:
                self.queue.task_done()

    async def _process_task(self, task: RolloutTask):
        """
        Executes a single micro-batch task.
        """
        async def _run_eval(rows_to_eval: Union[EvaluationRow, List[EvaluationRow]]):
            """Background evaluation task."""
            rollout_id = rows_to_eval[0].execution_metadata.rollout_id if isinstance(rows_to_eval, list) else rows_to_eval.execution_metadata.rollout_id
            experiment_id = rows_to_eval[0].execution_metadata.experiment_id if isinstance(rows_to_eval, list) else rows_to_eval.execution_metadata.experiment_id
            run_id = rows_to_eval[0].execution_metadata.run_id if isinstance(rows_to_eval, list) else rows_to_eval.execution_metadata.run_id
            eval_res = None

            # Track active eval
            async with self.active_evals_lock:
                self.active_evals += 1
                if self.eval_pbar:
                    self.eval_pbar.set_postfix_str(f"active={self.active_evals}")

            start_time = time.perf_counter()
            
            try:
                async with self.eval_sem:
                    async with rollout_logging_context(
                        rollout_id or "",
                        experiment_id=experiment_id,
                        run_id=run_id,
                    ):
                        if isinstance(rows_to_eval, list):
                            eval_res = await execute_pytest_with_exception_handling(
                                test_func=self.eval_executor,
                                evaluation_test_kwargs=self.evaluation_test_kwargs,
                                processed_dataset=rows_to_eval,
                            )
                        else:
                            eval_res = await execute_pytest_with_exception_handling(
                                test_func=self.eval_executor,
                                evaluation_test_kwargs=self.evaluation_test_kwargs,
                                processed_row=rows_to_eval,
                            )
                eval_duration = time.perf_counter() - start_time
                
                # Set eval_duration_seconds BEFORE buffer writes to ensure it's included in serialization
                if isinstance(eval_res, list):
                    for row in eval_res:
                        row.execution_metadata.eval_duration_seconds = eval_duration
                else:
                    eval_res.execution_metadata.eval_duration_seconds = eval_duration
                
                # push result to the output buffer
                if self.output_buffer:
                    if isinstance(eval_res, list):
                        for row in eval_res:
                            self._post_process_result(row)
                            await self.output_buffer.add_result(row)
                    else:
                        self._post_process_result(eval_res)
                        await self.output_buffer.add_result(eval_res)
                    
                if isinstance(eval_res, list):
                    for row in eval_res:
                        self.results.append(row)
                else:
                    self.results.append(eval_res)
                return eval_res
            finally:
                # Always update progress bar (handles both success and failure cases)
                if self.eval_pbar:
                    self.eval_pbar.update(1)
                # Decrement active eval counter
                async with self.active_evals_lock:
                    self.active_evals -= 1
                    if self.eval_pbar:
                        self.eval_pbar.set_postfix_str(f"active={self.active_evals}")

        # 1. Prepare Config & Row for this micro-batch
        current_batch_rows = []
        for run_idx in task.run_indices:
            row_copy = task.row.model_copy(deep=True)
            
            row_copy.execution_metadata.run_id = generate_id()
            row_copy.execution_metadata.rollout_id = generate_id()
            if row_copy.execution_metadata.extra is None:
                row_copy.execution_metadata.extra = {}
            row_copy.execution_metadata.extra["run_index"] = run_idx
            
            # Inject Speculation History
            if ENABLE_SPECULATION and task.history:
                cp = row_copy.input_metadata.completion_params
                max_tokens = cp.get("max_tokens", 2048)
                # Ensure safe dict access
                if not isinstance(cp, dict): 
                    cp = {}
                # Need to check and initialize nested dicts
                extra_body = cp.get("extra_body")
                if extra_body is None or not isinstance(extra_body, dict):
                    extra_body = {}
                # for speculation, see
                # https://docs.fireworks.ai/guides/predicted-outputs
                # https://platform.openai.com/docs/guides/predicted-outputs?lang=python
                extra_body["prediction"] = {"type": "content", "content": " ".join(task.history)[:max_tokens]}
                cp["extra_body"] = extra_body
                row_copy.input_metadata.completion_params = cp
            
            current_batch_rows.append((run_idx, row_copy))
            self.active_logger.log(row_copy)
        

        # 2. Execute Rollout
        batch_results: List[EvaluationRow] = []
        if current_batch_rows:
            for idx, row in current_batch_rows:
                # Track this rollout as active
                async with self.active_rollouts_lock:
                    self.active_rollouts[task.row_index].add(idx)
                    await self._update_rollout_pbar_postfix()
                
                try:
                    async for result_row in rollout_processor_with_retry(
                        self.rollout_processor, [row], task.config, idx, disable_tqdm=True
                    ):
                        batch_results.append(result_row)
                        
                        # Update rollout progress bar
                        if self.rollout_pbar:
                            self.rollout_pbar.update(1)
                        
                        # in pointwise, we start evaluation immediately
                        if self.mode == "pointwise":
                            t = asyncio.create_task(_run_eval(result_row))
                            self.background_tasks.add(t)
                            t.add_done_callback(self.background_tasks.discard)
                finally:
                    # Remove from active tracking
                    async with self.active_rollouts_lock:
                        self.active_rollouts[task.row_index].discard(idx)
                        if not self.active_rollouts[task.row_index]:
                            del self.active_rollouts[task.row_index]
                        await self._update_rollout_pbar_postfix()
        
        # 3. Evaluate and Collect History
        current_batch_history_updates = []
        # Extract history from rollout results (assuming eval doesn't change content needed for history)
        for res in batch_results:
            last_msg = res.last_assistant_message()
            if last_msg and last_msg.content:
                content = last_msg.content
                current_batch_history_updates.append(str(content))
            else:
                current_batch_history_updates.append("")

        # in groupwise, we send all rows to evaluator in one go when the whole group is complete
        if self.mode == "groupwise":
            self.groups_buffer[task.row_index].extend(batch_results)
            if len(self.groups_buffer[task.row_index]) >= self.rollout_n: 
                 full_group = self.groups_buffer.pop(task.row_index)
                 t = asyncio.create_task(_run_eval(full_group))
                 self.background_tasks.add(t)
                 t.add_done_callback(self.background_tasks.discard)

        # 4. Schedule Next Micro-batch (High Priority)
        last_run_idx = task.run_indices[-1] if task.run_indices else -1
        next_start = last_run_idx + 1
        
        if next_start < self.rollout_n:
            next_end = min(next_start + self.in_group_minibatch_size, self.rollout_n)
            next_indices = list(range(next_start, next_end))
            new_history = task.history + current_batch_history_updates
            
            # Priority 0 (High) to ensure we finish this sample ASAP
            new_priority = (0, task.row_index)
            
            new_task = RolloutTask(
                priority=new_priority,
                row=task.row,
                run_indices=next_indices,
                config=task.config,
                row_index=task.row_index,
                history=new_history
            )
            self.queue.put_nowait(new_task)

    def _format_active_rollouts(self) -> str:
        """Format active rollouts for display in progress bar."""
        if not self.active_rollouts:
            return ""
        
        # Show active rows and their run indices
        parts = []
        for row_idx in sorted(self.active_rollouts.keys())[:5]:  # Limit to 5 rows to keep it readable
            runs = sorted(self.active_rollouts[row_idx])
            if runs:
                runs_str = ",".join(str(r) for r in runs[:3])  # Show up to 3 run indices
                if len(runs) > 3:
                    runs_str += f"+{len(runs)-3}"
                parts.append(f"r{row_idx}:[{runs_str}]")
        
        if len(self.active_rollouts) > 5:
            parts.append(f"+{len(self.active_rollouts)-5} more")
        
        return " | ".join(parts)
    
    async def _update_rollout_pbar_postfix(self):
        """Update the rollout progress bar postfix with active tasks info."""
        if self.rollout_pbar:
            active_count = sum(len(runs) for runs in self.active_rollouts.values())
            self.rollout_pbar.set_postfix_str(
                f"active={active_count} {self._format_active_rollouts()}"
            )

    def _post_process_result(self, res: EvaluationRow):
        """
        Process evaluation result: update cost metrics, status, and log.
        """
        add_cost_metrics(res)
        if res.eval_metadata is not None:
            if res.rollout_status.is_error():
                res.eval_metadata.status = Status.error(
                    res.rollout_status.message, res.rollout_status.details
                )
            elif not (
                res.eval_metadata.status and res.eval_metadata.status.code != Status.Code.RUNNING
            ):
                res.eval_metadata.status = Status.eval_finished()
        
        if os.getenv("EP_DEBUG_SERIALIZATION", "0").strip() == "1":
            try:
                preview = [
                    {
                        "role": m.role,
                        "len": len(m.content or "") if isinstance(m.content, str) else None,
                        "tool_calls": len(m.tool_calls or [])
                        if hasattr(m, "tool_calls") and isinstance(m.tool_calls, list)
                        else 0,
                        "tool_call_id": getattr(m, "tool_call_id", None),
                        "name": getattr(m, "name", None),
                    }
                    for m in res.messages
                ]
                print("[EP-Log] Row messages:", preview)
            except Exception:
                pass
        self.active_logger.log(res)

    async def run(self, dataset: List[EvaluationRow], num_runs: int, base_config: RolloutProcessorConfig):
        self.num_runs = num_runs
        
        # Calculate totals for progress bars
        total_rollouts = len(dataset) * num_runs
        # In pointwise mode: 1 eval per rollout; in groupwise mode: 1 eval per dataset row
        total_evals = total_rollouts if self.mode == "pointwise" else len(dataset)
        
        # Initialize progress bars
        self.rollout_pbar = async_tqdm(
            total=total_rollouts,
            desc="🚀 Rollouts",
            unit="row",
            position=0,
            leave=True,
            colour="cyan",
        )
        self.eval_pbar = async_tqdm(
            total=total_evals,
            desc="📊 Evals",
            unit="eval",
            position=1,
            leave=True,
            colour="green",
        )
        
        try:
            # 1. Schedule initial tasks
            await self.schedule_dataset(dataset, base_config)
            
            # 2. Start Workers
            # If we have separate limits, we need enough workers to saturate both stages
            num_workers = self.max_concurrent_rollouts

            workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
            
            # 3. Wait for completion
            await self.queue.join()
            
            # Wait for background evaluations to finish
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # 4. Cleanup
            for w in workers:
                w.cancel()
            
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
        finally:
            # Close progress bars
            if self.rollout_pbar:
                self.rollout_pbar.close()
            if self.eval_pbar:
                self.eval_pbar.close()
            
        # Return collected results
        return self.results

async def execute_priority_rollouts(
    dataset: List[EvaluationRow],
    num_runs: int,
    rollout_processor: RolloutProcessor,
    config: RolloutProcessorConfig,
    max_concurrent_rollouts: int,
    active_logger: DatasetLogger,
    eval_executor: TestFunction,
    max_concurrent_evaluations: int = 96,
    mode: str = "pointwise",
    micro_batch_data_buffer: Optional[MicroBatchDataBuffer] = None,
    evaluation_test_kwargs: Dict[str, Any] = {},
):
    scheduler = PriorityRolloutScheduler(
        rollout_processor=rollout_processor,
        max_concurrent_rollouts=max_concurrent_rollouts,
        active_logger=active_logger,
        eval_executor=eval_executor,
        output_buffer=micro_batch_data_buffer,
        max_concurrent_evaluations=max_concurrent_evaluations,
        rollout_n=num_runs,
        mode=mode,
        in_group_minibatch_size=(num_runs // 2),
        evaluation_test_kwargs=evaluation_test_kwargs,
    )
    return await scheduler.run(dataset, num_runs, config)
