import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Dict, Optional, Union, Awaitable

from eval_protocol.models import EvaluationRow, Status
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.evaluation_test_utils import rollout_processor_with_retry
from eval_protocol.pytest.buffer import MiniBatchDataBuffer
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.human_id import generate_id

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
        eval_executor: Callable[[Union[EvaluationRow, List[EvaluationRow]]], Awaitable[Union[EvaluationRow, List[EvaluationRow]]]], # Callback to run evaluation
        mini_batch_data_buffer: Optional[MiniBatchDataBuffer] = None,
        max_concurrent_evaluations: Optional[int] = None,
    ):
        self.rollout_processor = rollout_processor
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.active_logger = active_logger
        self.eval_executor = eval_executor
        self.mini_batch_data_buffer = mini_batch_data_buffer
        
        # Priority Queue: Stores RolloutTask
        self.queue: asyncio.PriorityQueue[RolloutTask] = asyncio.PriorityQueue()
        
        # Concurrency Control
        self.rollout_sem = asyncio.Semaphore(max_concurrent_rollouts)
        self.eval_sem = asyncio.Semaphore(max_concurrent_evaluations) if max_concurrent_evaluations else None
        
        self.num_runs = 0
        self.micro_batch_size = 0

    async def schedule_dataset(
        self,
        dataset: List[EvaluationRow],
        base_config: RolloutProcessorConfig,
    ):
        """
        Populates the queue with initial tasks (the first micro-batch for each sample).
        """
        for i, row in enumerate(dataset):
            # Calculate ranges for the first micro-batch
            batch_start = 0
            # Ensure micro_batch_size is at least 1 to avoid infinite loop or stuck tasks
            safe_batch_size = self.micro_batch_size if self.micro_batch_size > 0 else self.num_runs
            batch_end = min(safe_batch_size, self.num_runs)
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
            try:
                # Get a task from the priority queue
                task: RolloutTask = await self.queue.get()
            except asyncio.QueueEmpty:
                break

            try:
                await self._process_task(task)
            except Exception as e:
                print(f"Error processing task for row {task.row.input_metadata.row_id}: {e}")
            finally:
                self.queue.task_done()

    async def _process_task(self, task: RolloutTask):
        """
        Executes a single micro-batch task.
        """
        # 1. Prepare Config & Row for this micro-batch
        current_batch_rows = []
        for run_idx in task.run_indices:
            row_copy = task.row.model_copy(deep=True)
            
            row_copy.execution_metadata.run_id = generate_id()
            row_copy.execution_metadata.rollout_id = generate_id()
            
            # Inject Speculation History
            if task.history:
                cp = row_copy.input_metadata.completion_params
                # Ensure safe dict access
                if not isinstance(cp, dict): 
                    cp = {}
                # Need to check and initialize nested dicts
                extra_body = cp.get("extra_body")
                if extra_body is None or not isinstance(extra_body, dict):
                    extra_body = {}
                
                extra_body["prediction"] = task.history
                cp["extra_body"] = extra_body
                row_copy.input_metadata.completion_params = cp
            
            current_batch_rows.append(row_copy)
            self.active_logger.log(row_copy)

        # 2. Execute Rollout
        batch_results: List[EvaluationRow] = []
        if task.run_indices:
            representative_run_idx = task.run_indices[0]
            
            async with self.rollout_sem:
                async for result_row in rollout_processor_with_retry(
                    self.rollout_processor, current_batch_rows, task.config, representative_run_idx
                ):
                    batch_results.append(result_row)
        
        # 3. Evaluate and Collect History
        current_batch_history_updates = []
        
        async def _run_eval():
            for res in batch_results:
                # Run Evaluation
                eval_res = await self.eval_executor(res)
                
                # Depending on the execution mode, eval_executor might return a single row or a list
                # For pointwise, it's a single row. For groupwise, it's a list.
                # Since PriorityScheduler processes a batch of single-turn rollouts, we expect single rows back
                # But to be safe and type-correct, we handle both.
                
                if isinstance(eval_res, list):
                    # Should not happen in pointwise mode which is typically used with this scheduler
                    # But if it does, we process each result
                    for r in eval_res:
                        if self.mini_batch_data_buffer:
                            await self.mini_batch_data_buffer.add_result(r)
                        
                        last_msg = r.last_assistant_message()
                        if last_msg and last_msg.content:
                            content = last_msg.content
                            if isinstance(content, list):
                                text_parts = [p["text"] for p in content if p["type"] == "text"]
                                current_batch_history_updates.append("".join(text_parts))
                            else:
                                current_batch_history_updates.append(str(content))
                        else:
                            current_batch_history_updates.append("")
                else:
                    if self.mini_batch_data_buffer:
                        await self.mini_batch_data_buffer.add_result(eval_res)

                    # Extract prediction for history
                    last_msg = eval_res.last_assistant_message()
                    if last_msg and last_msg.content:
                        content = last_msg.content
                        if isinstance(content, list):
                            text_parts = [p["text"] for p in content if p["type"] == "text"]
                            current_batch_history_updates.append("".join(text_parts))
                        else:
                            current_batch_history_updates.append(str(content))
                    else:
                        current_batch_history_updates.append("") # Empty string for failed turns

        if self.eval_sem:
            async with self.eval_sem:
                await _run_eval()
        else:
            await _run_eval()

        # 4. Schedule Next Micro-batch (High Priority)
        last_run_idx = task.run_indices[-1]
        next_start = last_run_idx + 1
        
        if next_start < self.num_runs:
            next_end = min(next_start + self.micro_batch_size, self.num_runs)
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

    async def run(self, dataset: List[EvaluationRow], num_runs: int, micro_batch_size: int, base_config: RolloutProcessorConfig):
        self.num_runs = num_runs
        self.micro_batch_size = micro_batch_size
        
        # 1. Schedule initial tasks
        await self.schedule_dataset(dataset, base_config)
        
        # 2. Start Workers
        # If we have separate limits, we need enough workers to saturate both stages
        num_workers = self.max_concurrent_rollouts
        if self.max_concurrent_evaluations:
            num_workers += self.max_concurrent_evaluations

        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        
        # 3. Wait for completion
        await self.queue.join()
        
        # 4. Cleanup
        for w in workers:
            w.cancel()
        
        # Ensure cancellation is complete
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
            
        # Return empty dict as we rely on side effects (streaming buffer)
        return {}

async def execute_priority_rollouts(
    dataset: List[EvaluationRow],
    num_runs: int,
    micro_batch_size: int,
    rollout_processor: RolloutProcessor,
    config: RolloutProcessorConfig,
    max_concurrent_rollouts: int,
    active_logger: DatasetLogger,
    eval_executor: Callable[[Union[EvaluationRow, List[EvaluationRow]]], Awaitable[Union[EvaluationRow, List[EvaluationRow]]]],
    mini_batch_data_buffer: Optional[MiniBatchDataBuffer] = None,
    max_concurrent_evaluations: Optional[int] = None,
):
    scheduler = PriorityRolloutScheduler(
        rollout_processor=rollout_processor,
        max_concurrent_rollouts=max_concurrent_rollouts,
        active_logger=active_logger,
        eval_executor=eval_executor,
        mini_batch_data_buffer=mini_batch_data_buffer,
        max_concurrent_evaluations=max_concurrent_evaluations
    )
    return await scheduler.run(dataset, num_runs, micro_batch_size, config)
