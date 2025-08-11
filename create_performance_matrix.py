import json
import os
import re
from collections import defaultdict

def create_performance_matrix():
    trajectory_dir = "trajectory_outputs"
    
    # Dictionary to store scores: task_id -> [run1_score, run2_score, ...]
    task_data = defaultdict(list)  # task_id -> list of (filename, score)
    
    # Process all JSON files in the directory
    for filename in os.listdir(trajectory_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(trajectory_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract row_id and evaluation score from JSON content
                if 'row_id' in data and 'evaluation' in data and 'score' in data['evaluation']:
                    row_id = data['row_id']
                    score = data['evaluation']['score']
                    task_data[row_id].append((filename, score))
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {filename}: {e}")
    
    # Create performance matrix
    # Matrix will be: rows = tasks, columns = runs
    performance_matrix = {}
    
    for task_id in sorted(task_data.keys()):
        files_and_scores = task_data[task_id]
        # Sort by filename (timestamp) to get consistent run ordering
        files_and_scores.sort(key=lambda x: x[0])
        
        # Extract just the scores for this task across all runs
        scores = [int(score) for _, score in files_and_scores[:8]]  # Take first 8 runs, convert to int
        performance_matrix[task_id] = scores
    
    # Print the performance matrix table
    print("Performance Matrix: Runs (columns) vs Tasks (rows)")
    print("=" * 80)
    
    # Header row
    header = "Task".ljust(15) + " | "
    for run_num in range(1, 9):
        header += f"Run{run_num}".center(6)
    print(header)
    print("-" * 80)
    
    # Data rows
    for task_id in sorted(performance_matrix.keys(), key=lambda x: int(x.split('_')[-1])):
        scores = performance_matrix[task_id]
        row = task_id.ljust(15) + " | "
        for score in scores:
            row += f"{score}".center(6)
        print(row)
    
    # Summary row showing totals per run
    print("-" * 80)
    totals_row = "TOTALS".ljust(15) + " | "
    for run_idx in range(8):
        total = sum(performance_matrix[task_id][run_idx] for task_id in performance_matrix.keys())
        totals_row += f"{total}".center(6)
    print(totals_row)
    
    print("\nSummary:")
    print("- 1 = Task succeeded")
    print("- 0 = Task failed")
    print("- Bottom row shows total successes per run")

if __name__ == "__main__":
    create_performance_matrix() 