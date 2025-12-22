import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Union

from eval_protocol.models import EvaluationRow, InputMetadata, ExecutionMetadata, EvaluateResult
from eval_protocol.pytest.priority_scheduler import PriorityRolloutScheduler, execute_priority_rollouts, RolloutTask, SampleState
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
    async def delayed_rollout(processor, rows, config, run_idx, **kwargs):
        await asyncio.sleep(0.01)
        for row in rows:
            yield row

    async def mock_eval(row):
        row.evaluation_result = EvaluateResult(score=1.0, is_score_valid=True)
        return row

    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=delayed_rollout):
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=2,
            active_logger=mock_logger,
            eval_executor=mock_eval,
            max_concurrent_evaluations=2,
            rollout_n=num_runs,
            in_group_minibatch_size=micro_batch_size
        )
        
        results = await scheduler.run(dataset, num_runs, base_config)
        
        assert len(results) == 5 * num_runs
        for res in results:
            assert res.evaluation_result is not None
            assert res.evaluation_result.score == 1.0


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

    async def mock_rollout_gen(processor, rows, config, run_idx, **kwargs):
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

    # Use a real async function for eval to work with execute_pytest properly
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
        
        # Mock processor instance (can be anything since we patched the wrapper)
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=max_rollouts,
            active_logger=mock_logger,
            eval_executor=mock_eval,
            max_concurrent_evaluations=max_evals,
            rollout_n=num_runs,
            in_group_minibatch_size=micro_batch_size
        )
        
        await scheduler.run(dataset, num_runs, base_config)
        
        # Verify limits were respected
        assert max_active_rollouts_seen <= max_rollouts, f"Rollout concurrency exceeded: {max_active_rollouts_seen} > {max_rollouts}"
        assert max_active_evals_seen <= max_evals, f"Eval concurrency exceeded: {max_active_evals_seen} > {max_evals}"
        
        # Verify everything ran
        # 10 rows * 1 run = 10 results
        assert len(scheduler.results) == 10

@pytest.mark.asyncio
async def test_priority_scheduling(
    mock_logger, mock_eval_executor, base_config
):
    """
    Test that subsequent micro-batches are prioritized.
    """
    dataset = [create_mock_row(f"row-{i}") for i in range(2)]
    num_runs = 2
    micro_batch_size = 1
    
    execution_order = []
    
    async def mock_rollout_gen(processor, rows, config, run_idx, **kwargs):
        row_id = rows[0].input_metadata.row_id
        execution_order.append(f"{row_id}_run_{run_idx}")
        for row in rows:
            yield row

    async def mock_eval(row):
        return row

    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=mock_rollout_gen):
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=1, # Force serial execution to test priority
            active_logger=mock_logger,
            eval_executor=mock_eval,
            max_concurrent_evaluations=1,
            rollout_n=num_runs,
            in_group_minibatch_size=micro_batch_size
        )
        
        await scheduler.run(dataset, num_runs, base_config)
        
        # Expected order: row-0_run_0, row-0_run_1, row-1_run_0, row-1_run_1
        # Note: Since row-0_run_0 finishes, it schedules row-0_run_1 with HIGH priority (0).
        # row-1_run_0 is in queue with LOW priority (1).
        # So row-0_run_1 should run before row-1_run_0.
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
    # Updated expectation: workers only scale with rollout concurrency now
    expected_workers = max_rollouts
    
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
                 sample_state = SampleState(
                     row=dataset[0],
                     row_index=0,
                     config=base_config,
                     history=[],
                     next_run_idx=0,
                     active_runs=0,
                     completed_runs=0,
                     lock=asyncio.Lock(),
                 )
                 task = RolloutTask(
                     priority=(1, i, 0),
                     sample_state=sample_state,
                     run_idx=0,
                     history_snapshot=[],
                 )
                 await self.queue.put(task)

    processor_instance = MagicMock()
    scheduler = InstrumentedScheduler(
        rollout_processor=processor_instance,
        max_concurrent_rollouts=max_rollouts,
        active_logger=mock_logger,
        eval_executor=mock_eval_executor,
        max_concurrent_evaluations=max_evals,
        rollout_n=1,
        in_group_minibatch_size=1
    )
    
    await scheduler.run(dataset, 1, base_config)
    
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

    async def mock_rollout_gen(processor, rows, config, run_idx, **kwargs):
        for row in rows:
            yield row
    
    with patch('eval_protocol.pytest.priority_scheduler.rollout_processor_with_retry', side_effect=mock_rollout_gen):
        processor_instance = MagicMock()
        
        scheduler = PriorityRolloutScheduler(
            rollout_processor=processor_instance,
            max_concurrent_rollouts=1,
            active_logger=mock_logger,
            eval_executor=mock_eval,
            max_concurrent_evaluations=1,
            mode="groupwise",
            rollout_n=num_runs,
            in_group_minibatch_size=micro_batch_size
        )
        
        results = await scheduler.run(dataset, num_runs, base_config)
        
        # Verify evaluation was called EXACTLY ONCE
    assert len(eval_calls) == 1, f"Expected 1 eval call, got {len(eval_calls)}"
    
    # Verify it was called with ALL 4 rows
    evaluated_rows = eval_calls[0]
    assert len(evaluated_rows) == 4, f"Expected 4 rows in group eval, got {len(evaluated_rows)}"
    
    # Verify results contains all 4 runs (returned from eval)
    # Note: eval returns a list of 4 rows. scheduler.results extends this list.
    assert len(results) == 4
