#!/usr/bin/env python3
"""
Test script for compute_fixed_set_mu_ci function with real dataset from input.txt.
Using the complex approach with proper task-based grouping.
"""

import re
from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci
from eval_protocol.models import EvaluationRow, EvaluateResult, InputMetadata

def parse_input_file(filename: str):
    """Parse the input.txt file and extract task data."""
    
    task_data = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match: Task: X | Trial: Y | Reward: ✅/❌ | Duration: Z | DB Match: YES/NO |
    pattern = r'Task: (\d+) \| Trial: (\d+) \| Reward: ([✅❌]) \| Duration: ([\d.]+)s \| DB Match: (YES|NO)'
    
    matches = re.findall(pattern, content)
    
    for task_id, trial_id, reward, duration, db_match in matches:
        task_num = int(task_id)
        trial_num = int(trial_id)
        success = 1.0 if reward == '✅' else 0.0
        duration_val = float(duration)
        
        task_data.append({
            'task_id': task_num,
            'trial_id': trial_num,
            'score': success,
            'duration': duration_val,
            'db_match': db_match == 'YES'
        })
    
    return task_data

def test_real_dataset_ci():
    """Test the complex approach on the real dataset from input.txt."""
    
    print("="*60)
    print("=== REAL DATASET ANALYSIS ===")
    
    # Parse the input file
    try:
        task_data = parse_input_file('input.txt')
    except FileNotFoundError:
        print("ERROR: input.txt file not found!")
        return
    
    if not task_data:
        print("ERROR: No data found in input.txt!")
        return
    
    print(f"Parsed {len(task_data)} entries from input.txt")
    
    # Get unique tasks and trials
    unique_tasks = set(entry['task_id'] for entry in task_data)
    unique_trials = set(entry['trial_id'] for entry in task_data)
    
    print(f"Tasks: {min(unique_tasks)} to {max(unique_tasks)} ({len(unique_tasks)} unique tasks)")
    print(f"Trials: {min(unique_trials)} to {max(unique_trials)} ({len(unique_trials)} trials per task)")
    
    # Calculate overall success rate
    total_successes = sum(entry['score'] for entry in task_data)
    overall_rate = total_successes / len(task_data)
    print(f"Overall success rate: {total_successes}/{len(task_data)} = {overall_rate:.3f} ({overall_rate*100:.1f}%)")
    
    # Create EvaluationRow objects with proper task grouping
    evaluation_rows = []
    for entry in task_data:
        # Use task_id as the row_id so multiple trials of the same task get grouped
        row_id = f"task_{entry['task_id']}"
        
        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(row_id=row_id)
        )
        
        # Set the evaluation result with the score
        row.evaluation_result = EvaluateResult(score=entry['score'])
        
        evaluation_rows.append(row)
    
    # Compute the confidence interval using the complex approach
    result_ci = compute_fixed_set_mu_ci(evaluation_rows)
    
    print(f"\nResult from compute_fixed_set_mu_ci:")
    print(f"result_ci = {result_ci}")
    
    if result_ci[0] is not None:
        mu_hat, ci_low, ci_high = result_ci
        print(f"\nDetailed results:")
        print(f"μ̂ (estimated mean): {mu_hat:.4f} ({mu_hat*100:.1f}%)")
        print(f"95% CI lower bound: {ci_low:.4f} ({ci_low*100:.1f}%)")
        print(f"95% CI upper bound: {ci_high:.4f} ({ci_high*100:.1f}%)")
        print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}] = [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
        print(f"Margin of error: ±{(ci_high - ci_low) / 2:.4f} (±{(ci_high - ci_low) / 2 * 100:.1f}%)")
        
        # Show per-task statistics
        task_stats = {}
        for entry in task_data:
            task_id = entry['task_id']
            if task_id not in task_stats:
                task_stats[task_id] = []
            task_stats[task_id].append(entry['score'])
        
        print(f"\nPer-task success rates (first 10 tasks):")
        for task_id in sorted(task_stats.keys())[:10]:
            scores = task_stats[task_id]
            success_rate = sum(scores) / len(scores)
            print(f"  Task {task_id:3d}: {sum(scores)}/{len(scores)} = {success_rate:.3f} ({success_rate*100:5.1f}%)")
        
        if len(task_stats) > 10:
            print(f"  ... and {len(task_stats) - 10} more tasks")
            
    else:
        print("Could not compute confidence interval (insufficient data)")

if __name__ == "__main__":
    test_real_dataset_ci() 