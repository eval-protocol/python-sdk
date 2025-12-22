import asyncio
import copy
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


@dataclass
class SampleState:
    """
    Tracks state for a single dataset sample across multiple runs.
    Enables streaming scheduling where each completed run immediately triggers the next.
    """
    row: EvaluationRow
    row_index: int
    config: RolloutProcessorConfig
    history: List[str] = field(default_factory=list)  # Accumulated history from completed runs
    next_run_idx: int = 0  # Next run index to schedule
    active_runs: int = 0  # Currently executing runs for this sample
    completed_runs: int = 0  # Total completed runs for this sample
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Protect state updates


@dataclass(order=True)
class RolloutTask:
    """
    Represents a single rollout task (one run for one sample).
    Priority tuple structure: (status, row_index, run_index)
      - status: 0 = High Priority (continuing a started sample)
                1 = Low Priority (starting a new sample)
      - row_index: Dataset order
      - run_index: Run order within sample
    """
    priority: tuple[int, int, int]
    
    # Payload (excluded from comparison)
    sample_state: SampleState = field(compare=False)
    run_idx: int = field(compare=False)  # Single run index for this task
    history_snapshot: List[str] = field(compare=False, default_factory=list)  # History at scheduling time

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
        in_group_minibatch_size: Optional[int] = None, # for one sample, how many runs to execute at the same time
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
        
        # Concurrency Control (rollout concurrency is handled by rollout_processor's semaphore)
        self.eval_sem = asyncio.Semaphore(max_concurrent_evaluations)
        
        # Results storage
        self.results: List[EvaluationRow] = [] # for backward compatibility reason, we save all results here to return
        self.groups_buffer: Dict[int, List[EvaluationRow]] = defaultdict(list) # buffer for group results. only flush to output buffer when a whole group is ready
        
        self.background_tasks = set() # run evaluations in the background asynchronously
        
        self.rollout_n = rollout_n
        if in_group_minibatch_size is None:
            if ENABLE_SPECULATION:
                in_group_minibatch_size = rollout_n // 2
            else:
                in_group_minibatch_size = rollout_n
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

        self.sample_states: Dict[int, SampleState] = {}

    async def schedule_dataset(
        self,
        dataset: List[EvaluationRow],
        base_config: RolloutProcessorConfig,
    ):
        """
        Populates the queue with initial tasks.
        For each sample, schedules up to in_group_minibatch_size concurrent runs.
        """
        for i, row in enumerate(dataset):
            # Create sample state
            sample_state = SampleState(
                row=row,
                row_index=i,
                config=base_config,
                history=[],
                next_run_idx=0,
                active_runs=0,
                completed_runs=0,
                lock=asyncio.Lock(),
            )
            self.sample_states[i] = sample_state
            
            # Schedule initial runs (up to in_group_minibatch_size)
            initial_runs = min(self.in_group_minibatch_size, self.rollout_n)
            for run_idx in range(initial_runs):
                # Initial priority: Low (1), ordered by dataset index, then run index
                priority = (1, i, run_idx)
                
                task = RolloutTask(
                    priority=priority,
                    sample_state=sample_state,
                    run_idx=run_idx,
                    history_snapshot=[],  # First runs have no history
                )
                self.queue.put_nowait(task)
                sample_state.next_run_idx = run_idx + 1
                sample_state.active_runs += 1

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
                logging.error(f"Error processing task for row {task.sample_state.row.input_metadata.row_id} run {task.run_idx}: {e}", exc_info=True)
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

        sample_state = task.sample_state
        run_idx = task.run_idx
        row_index = sample_state.row_index
        
        # Rollout concurrency is handled by rollout_processor's internal semaphore
        # 1. Prepare row for this single run
        row_copy = sample_state.row.model_copy(deep=True)
        row_copy.execution_metadata.run_id = generate_id()
        row_copy.execution_metadata.rollout_id = generate_id()
        if row_copy.execution_metadata.extra is None:
            row_copy.execution_metadata.extra = {}
        row_copy.execution_metadata.extra["run_index"] = run_idx
        
        # Make a copy of config for this specific run (to inject per-run speculation)
        run_config = sample_state.config
        
        # Inject Speculation History into config.completion_params (use snapshot from when task was scheduled)
        if ENABLE_SPECULATION and task.history_snapshot:
            # Deep copy to avoid concurrent mutation of shared nested dicts
            cp = copy.deepcopy(sample_state.config.completion_params) if sample_state.config.completion_params else {}
            max_tokens = cp.get("max_tokens", 2048)
            if "extra_body" not in cp:
                cp["extra_body"] = {}
            
            cp["extra_body"]["prediction"] = {"type": "content", "content": " ".join(task.history_snapshot)[:max_tokens]}
            
            # Create a new config with the modified completion_params (copy all fields)
            base_config = sample_state.config
            run_config = RolloutProcessorConfig(
                completion_params=cp,
                mcp_config_path=base_config.mcp_config_path,
                semaphore=base_config.semaphore,
                server_script_path=base_config.server_script_path,
                steps=base_config.steps,
                logger=base_config.logger,
                kwargs=base_config.kwargs,
                exception_handler_config=base_config.exception_handler_config,
                post_processor=base_config.post_processor,
            )
        
        self.active_logger.log(row_copy)
        
        # 2. Track this rollout as active
        async with self.active_rollouts_lock:
            self.active_rollouts[row_index].add(run_idx)
            await self._update_rollout_pbar_postfix()
        
        # 3. Execute the rollout
        result_row: Optional[EvaluationRow] = None
        start_time = time.perf_counter()
        try:
            async for result in rollout_processor_with_retry(
                self.rollout_processor, [row_copy], run_config, run_idx, disable_tqdm=True
            ):
                result_row = result
                result_row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time
                
                # Update rollout progress bar
                if self.rollout_pbar:
                    self.rollout_pbar.update(1)
                
                # In pointwise mode, start evaluation immediately
                if self.mode == "pointwise":
                    t = asyncio.create_task(_run_eval(result_row))
                    self.background_tasks.add(t)
                    t.add_done_callback(self.background_tasks.discard)
        finally:
            # Remove from active tracking
            async with self.active_rollouts_lock:
                self.active_rollouts[row_index].discard(run_idx)
                if not self.active_rollouts[row_index]:
                    del self.active_rollouts[row_index]
                await self._update_rollout_pbar_postfix()
        
            # 4. Update sample state and schedule next run (streaming)
            # Must be in finally to ensure state is updated even on exception
            async with sample_state.lock:
                sample_state.active_runs -= 1
                sample_state.completed_runs += 1
                
                # Extract history from this run's result
                if result_row:
                    last_msg = result_row.last_assistant_message()
                    if last_msg and last_msg.content:
                        sample_state.history.append(str(last_msg.content))
                    else:
                        sample_state.history.append("")
                
                # In groupwise mode, buffer results
                if self.mode == "groupwise":
                    if result_row:
                        self.groups_buffer[row_index].append(result_row)
                    # Check if all runs for this sample are complete
                    if sample_state.completed_runs >= self.rollout_n:
                        full_group = self.groups_buffer.pop(row_index, [])
                        if full_group:
                            t = asyncio.create_task(_run_eval(full_group))
                            self.background_tasks.add(t)
                            t.add_done_callback(self.background_tasks.discard)
                
                # Schedule next run if:
                # 1. There are more runs to do
                # 2. We haven't hit in_group_minibatch_size concurrent runs for this sample
                if (sample_state.next_run_idx < self.rollout_n and 
                    sample_state.active_runs < self.in_group_minibatch_size):
                    
                    next_run_idx = sample_state.next_run_idx
                    sample_state.next_run_idx += 1
                    sample_state.active_runs += 1
                    
                    # High priority (0) to finish this sample ASAP
                    # Use current accumulated history for speculation
                    priority = (0, row_index, next_run_idx)
                    
                    new_task = RolloutTask(
                        priority=priority,
                        sample_state=sample_state,
                        run_idx=next_run_idx,
                        history_snapshot=list(sample_state.history),  # Snapshot current history
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
            # With semaphore-based concurrency control, workers can be equal to max_concurrent_rollouts
            # The semaphore will limit actual concurrent executions
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
        evaluation_test_kwargs=evaluation_test_kwargs,
    )
    return await scheduler.run(dataset, num_runs, config)
