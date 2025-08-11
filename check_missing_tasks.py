#!/usr/bin/env python3
"""
Script to analyze input.txt and find missing task/trial combinations.
"""

import re
from collections import defaultdict

def analyze_missing_tasks():
    """Analyze input.txt to find missing task/trial combinations."""
    
    # Parse the input file
    with open('input.txt', 'r') as f:
        content = f.read()
    
    # Pattern to match: Task: X | Trial: Y | Reward: ✅/❌ | Duration: Z | DB Match: YES/NO |
    pattern = r'Task: (\d+) \| Trial: (\d+) \| Reward: ([✅❌]) \| Duration: ([\d.]+)s \| DB Match: (YES|NO)'
    
    matches = re.findall(pattern, content)
    
    # Track which task/trial combinations we found
    task_trials = defaultdict(set)
    
    for task_id, trial_id, reward, duration, db_match in matches:
        task_num = int(task_id)
        trial_num = int(trial_id)
        task_trials[task_num].add(trial_num)
    
    print(f"Found {len(matches)} total entries")
    print(f"Found {len(task_trials)} unique tasks")
    
    # Find the range of tasks and trials
    all_tasks = set(task_trials.keys())
    all_trials_found = set()
    for trials in task_trials.values():
        all_trials_found.update(trials)
    
    min_task, max_task = min(all_tasks), max(all_tasks)
    min_trial, max_trial = min(all_trials_found), max(all_trials_found)
    
    print(f"Task range: {min_task} to {max_task}")
    print(f"Trial range: {min_trial} to {max_trial}")
    
    # Expected: all tasks should have trials 0-7 (8 trials each)
    expected_trials = set(range(8))  # {0, 1, 2, 3, 4, 5, 6, 7}
    
    # Find tasks with missing trials
    incomplete_tasks = []
    missing_combinations = []
    
    # Check all tasks in the range
    for task_id in range(min_task, max_task + 1):
        if task_id not in task_trials:
            # Task completely missing
            incomplete_tasks.append((task_id, "COMPLETELY MISSING"))
            for trial_id in expected_trials:
                missing_combinations.append((task_id, trial_id))
        else:
            # Task exists but may be missing some trials
            found_trials = task_trials[task_id]
            missing_trials = expected_trials - found_trials
            
            if missing_trials:
                incomplete_tasks.append((task_id, f"Missing trials: {sorted(missing_trials)}"))
                for trial_id in missing_trials:
                    missing_combinations.append((task_id, trial_id))
    
    # Report results
    print(f"\n{'='*60}")
    print("TASK COMPLETION ANALYSIS")
    print(f"{'='*60}")
    
    complete_tasks = len(task_trials) - len(incomplete_tasks)
    print(f"Complete tasks (8/8 trials): {complete_tasks}")
    print(f"Incomplete tasks: {len(incomplete_tasks)}")
    print(f"Total missing combinations: {len(missing_combinations)}")
    
    if incomplete_tasks:
        print(f"\nINCOMPLETE TASKS:")
        for task_id, issue in incomplete_tasks:
            print(f"  Task {task_id:3d}: {issue}")
    
    if missing_combinations:
        print(f"\nMISSING TASK/TRIAL COMBINATIONS:")
        # Group by task for better readability
        missing_by_task = defaultdict(list)
        for task_id, trial_id in missing_combinations:
            missing_by_task[task_id].append(trial_id)
        
        for task_id in sorted(missing_by_task.keys()):
            trials = sorted(missing_by_task[task_id])
            print(f"  Task {task_id:3d}: Trials {trials}")
    
    # Show tasks with full coverage
    print(f"\nTASKS WITH COMPLETE COVERAGE (8/8 trials):")
    complete_task_ids = []
    for task_id in sorted(task_trials.keys()):
        if len(task_trials[task_id]) == 8:
            complete_task_ids.append(task_id)
    
    print(f"  {len(complete_task_ids)} tasks: {complete_task_ids[:20]}{'...' if len(complete_task_ids) > 20 else ''}")
    
    return missing_combinations

if __name__ == "__main__":
    missing = analyze_missing_tasks()
    print(f"\nSUMMARY: {len(missing)} missing task/trial combinations found.") 