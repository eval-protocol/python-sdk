import asyncio
import os
from collections import defaultdict
from typing import List, Dict

from eval_protocol.models import EvaluationRow

class MicroBatchDataBuffer:
    """
    Buffers evaluation results and writes them to disk in minibatches.
    Waits for all runs of a sample to complete before considering it ready and flush to disk.
    """
    def __init__(self, num_runs: int, batch_size: int, output_path_template: str):
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.output_path_template = output_path_template
        self.pending_samples: Dict[str, List[EvaluationRow]] = defaultdict(list)  # row_id -> list[EvaluationRow]
        self.completed_samples_buffer: List[List[EvaluationRow]] = []  # List[List[EvaluationRow]]
        self.batch_index = 0
        self.lock = asyncio.Lock()

    async def add_result(self, row: EvaluationRow):
        """
        Add a single evaluation result.
        Thread-safe/Coroutine-safe.
        """
        async with self.lock:
            row_id = row.input_metadata.row_id
            if not row_id:
                # Should not happen in valid EP workflow, unique row_id is required to group things together properly
                return
            
            self.pending_samples[row_id].append(row)
            
            if len(self.pending_samples[row_id]) >= self.num_runs:
                # Sample completed (all runs finished)
                completed_rows = self.pending_samples.pop(row_id)
                self.completed_samples_buffer.append(completed_rows)
                
                if len(self.completed_samples_buffer) >= self.batch_size:
                    await self._flush_unsafe()

    async def _flush_unsafe(self):
        """
        not thread safe, assumes lock is held by called
        """
        if not self.completed_samples_buffer:
            return

        if "{index}" in self.output_path_template:
            output_path = self.output_path_template.format(index=self.batch_index)
            mode = "w"
        else:
            output_path = self.output_path_template
            mode = "a"  # Append if no index placeholder

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write flattened rows
        with open(output_path, mode) as f:
            for sample_rows in self.completed_samples_buffer:
                for row in sample_rows:
                    f.write(row.model_dump_json() + "\n")
        
        self.completed_samples_buffer = []
        self.batch_index += 1

    async def close(self):
        """
        Flush any remaining samples in the buffer.
        """
        async with self.lock:
            # Also flush pending (incomplete) samples to avoid data loss
            if self.pending_samples:
                for rows in self.pending_samples.values():
                    self.completed_samples_buffer.append(rows)
                self.pending_samples.clear()

            if self.completed_samples_buffer:
                await self._flush_unsafe()

