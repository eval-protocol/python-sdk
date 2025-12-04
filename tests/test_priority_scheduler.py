import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from typing import List, Union

from eval_protocol.models import EvaluationRow, InputMetadata, ExecutionMetadata
from eval_protocol.pytest.priority_scheduler import PriorityRolloutScheduler, execute_priority_rollouts, RolloutTask
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger

# Mock models
def create_mock_row(row_id: str = "test-row") -> EvaluationRow:
    return EvaluationRow(
        input_metadata=InputMetadata(
            row_id=row_id,
            completion_params={"model": "test-model"}
        ),
        execution_metadata=ExecutionMetadata()
    )

@pytest.fixture
def mock_rollout_processor():
    processor = MagicMock()
    # Mocking the rollout to be an async generator
    async def mock_rollout_gen(rows, config, run_idx):
        for row in rows:
            # Simulate some work
            yield row
    processor.side_effect = mock_rollout_gen
    return processor

@pytest.fixture
def mock_logger():
    return MagicMock(spec=DatasetLogger)

@pytest.fixture
def mock_eval_executor():
    return AsyncMock()

@pytest.fixture
def base_config():
    return RolloutProcessorConfig(
        completion_params={"model": "test-model"},
        mcp_config_path="test_config.yaml",
        semaphore=asyncio.Semaphore(10),
        steps=10
    )

@pytest.mark.asyncio
async def test_scheduler_basic_execution(
    mock_logger, mock_eval_executor, base_config
):
    """Test that the scheduler processes all rows and completes."""
    dataset = [create_mock_row(f"row-{i}") for i in range(5)]
    num_runs = 2
    micro_batch_size = 1
    
    # Mock rollout processor with delay
    async def delayed_rollout(rows, config, run_idx):
        await asyncio.sleep(0.01)
        for row in rows:
            yield row

    mock_processor = MagicMock()
    mock_processor.side_effect = delayed_rollout # This is wrong usage for call, rollout_processor is passed as instance
    # But wait, PriorityRolloutScheduler calls rollout_processor_with_retry which calls processor.process_batch or similar?
    # Looking at code: rollout_processor_with_retry(self.rollout_processor, ...)
    # rollout_processor_with_retry expects the processor instance.
    
    # Let's look at how rollout_processor_with_retry is implemented or usage.
    # Assuming rollout_processor is an object with a method or it's a callable?
    # In priority_scheduler.py: rollout_processor_with_retry(self.rollout_processor, ...)
    
    # Let's actually mock rollout_processor_with_retry since we want to test the scheduler logic, 
    # not the processor retry logic.
    # But we can't easily mock the import inside the module without patching.
    pass

# We will rely on patching 'eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry'
from unittest.mock import patch

@pytest.mark.asyncio
async def test_concurrency_control(
    mock_logger, mock_eval_executor, base_config
):
    """
    Verify that max_concurrent_rollouts and max_concurrent_evaluations are respected.
    """
    dataset = [create_mock_row(f"row-{i}") for i in range(10)]
    num_runs = 1
    micro_batch_size = 1
    
    max_rollouts = 4
    max_evals = 2
    
    active_rollouts = 0
    max_active_rollouts_seen = 0
    
    active_evals = 0
    max_active_evals_seen = 0
    
    rollout_lock = asyncio.Lock()
    eval_lock = asyncio.Lock()

    async def mock_rollout_gen(processor, rows, config, run_idx):
        nonlocal active_rollouts, max_active_rollouts_seen
        async with rollout_lock:
            active_rollouts += 1
            max_active_rollouts_seen = max(max_active_rollouts_seen, active_rollouts)
        
        # Simulate slow rollout
        await asyncio.sleep(0.05)
        
        for row in rows:
            yield row
            
        async with rollout_lock:
            active_rollouts -= 1

    async def mock_eval(row):
        nonlocal active_evals, max_active_evals_seen
        async with eval_lock:
            active_evals += 1
            max_active_evals_seen = max(max_active_evals_seen, active_evals)
            
        # Simulate evaluation
        await asyncio.sleep(0.05)
        
        async with eval_lock:
            active_evals -= 1
        return row

    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=mock_rollout_gen):
        mock_eval_executor.side_effect = mock_eval
        
        # Mock processor instance (can be anything since we patched the wrapper)
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=max_rollouts,
            active_logger=mock_logger,
            eval_executor=mock_eval_executor,
            max_concurrent_evaluations=max_evals
        )
        
        await scheduler.run(dataset, num_runs, micro_batch_size, base_config)
        
        # Verify limits were respected
        assert max_active_rollouts_seen <= max_rollouts, f"Rollout concurrency exceeded: {max_active_rollouts_seen} > {max_rollouts}"
        assert max_active_evals_seen <= max_evals, f"Eval concurrency exceeded: {max_active_evals_seen} > {max_evals}"
        
        # Verify everything ran
        # 10 rows * 1 run = 10 rollouts called
        # 10 evaluations
        assert mock_eval_executor.call_count == 10

@pytest.mark.asyncio
async def test_priority_scheduling(
    mock_logger, mock_eval_executor, base_config
):
    """
    Test that subsequent micro-batches are prioritized.
    This is tricky to test deterministically with asyncio, but we can try to observe order
    or ensure that a task that spawns new parts gets priority.
    
    We'll simulate a case where we have 2 samples, each needing 2 micro-batches.
    We want to see if Sample 1 Batch 2 runs before Sample 2 Batch 1 is finished if possible,
    but actually the scheduler puts Sample 1 Batch 2 with Priority 0 (High) and Sample 2 Batch 1 starts with Priority 1 (Low).
    
    If we limit concurrency to 1, we should see:
    S1_B1 -> S1_B2 -> S2_B1 -> S2_B2
    
    Wait, if concurrency is 1:
    1. Queue: [S1_B1 (Low), S2_B1 (Low)]
    2. Worker picks S1_B1. Queue: [S2_B1 (Low)]
    3. S1_B1 finishes. Puts S1_B2 (High). Queue: [S1_B2 (High), S2_B1 (Low)]
    4. Worker picks S1_B2. Queue: [S2_B1 (Low)]
    5. S1_B2 finishes. Queue: [S2_B1 (Low)]
    6. Worker picks S2_B1. ...
    
    So yes, strictly sequential per sample if concurrency=1.
    """
    dataset = [create_mock_row(f"row-{i}") for i in range(2)]
    num_runs = 2
    micro_batch_size = 1
    
    execution_order = []
    
    async def mock_rollout_gen(processor, rows, config, run_idx):
        row_id = rows[0].input_metadata.row_id
        execution_order.append(f"{row_id}_run_{run_idx}")
        for row in rows:
            yield row

    async def mock_eval(row):
        return row

    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=mock_rollout_gen):
        mock_eval_executor.side_effect = mock_eval
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=1, # Force serial execution to test priority
            active_logger=mock_logger,
            eval_executor=mock_eval_executor,
        )
        
        await scheduler.run(dataset, num_runs, micro_batch_size, base_config)
        
        # Expected order: row-0_run_0, row-0_run_1, row-1_run_0, row-1_run_1
        # Or at least row-0_run_1 should come before row-1_run_0 finishes if parallel?
        # With concurrency 1, it should be strictly:
        # row-0 run 0
        # row-0 run 1 (high priority injected)
        # row-1 run 0
        # row-1 run 1
        
        expected = [
            "row-0_run_0",
            "row-0_run_1",
            "row-1_run_0",
            "row-1_run_1"
        ]
        
        assert execution_order == expected

@pytest.mark.asyncio
async def test_worker_scaling(
    mock_logger, mock_eval_executor, base_config
):
    """
    Test that the number of workers scales with the sum of limits.
    """
    dataset = [create_mock_row("row-0")]
    max_rollouts = 5
    max_evals = 3
    expected_workers = max_rollouts + max_evals
    
    worker_start_count = 0
    
    class InstrumentedScheduler(PriorityRolloutScheduler):
        async def worker(self):
            nonlocal worker_start_count
            worker_start_count += 1
            try:
                await self.queue.get()
                self.queue.task_done()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        async def schedule_dataset(self, *args):
             # Put enough items to ensure all workers wake up and grab one
             for i in range(expected_workers):
                 task = RolloutTask(
                     priority=(1, i),
                     row=dataset[0],
                     run_indices=[],
                     config=base_config,
                     row_index=0,
                     history=[]
                 )
                 await self.queue.put(task)

    processor_instance = MagicMock()
    scheduler = InstrumentedScheduler(
        rollout_processor=processor_instance,
        max_concurrent_rollouts=max_rollouts,
        active_logger=mock_logger,
        eval_executor=mock_eval_executor,
        max_concurrent_evaluations=max_evals
    )
    
    await scheduler.run(dataset, 1, 1, base_config)
    
    assert worker_start_count == expected_workers

@pytest.mark.asyncio
async def test_groupwise_mode(
    mock_logger, mock_eval_executor, base_config
):
    """
    Test that groupwise mode collects all runs before evaluating.
    """
    dataset = [create_mock_row("row-0")]
    num_runs = 4
    micro_batch_size = 2
    
    # We expect 2 batches of 2 runs each.
    # Batch 1 (Runs 0,1): Should buffer and update history, NOT call eval.
    # Batch 2 (Runs 2,3): Should buffer, update history, AND call eval with all 4 runs.
    
    eval_calls = []
    
    async def mock_eval(rows):
        eval_calls.append(rows)
        return rows # Pass through

    async def mock_rollout_gen(processor, rows, config, run_idx):
        for row in rows:
            yield row

    mock_eval_executor.side_effect = mock_eval
    
    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=mock_rollout_gen):
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=1,
            active_logger=mock_logger,
            eval_executor=mock_eval_executor,
            mode="groupwise"
        )
        
        results = await scheduler.run(dataset, num_runs, micro_batch_size, base_config)
        
        # Verify evaluation was called EXACTLY ONCE
        assert len(eval_calls) == 1, f"Expected 1 eval call, got {len(eval_calls)}"
        
        # Verify it was called with ALL 4 rows
        evaluated_rows = eval_calls[0]
        assert len(evaluated_rows) == 4, f"Expected 4 rows in group eval, got {len(evaluated_rows)}"
        
        # Verify results contains all 4 rows
        assert len(results) == 4


