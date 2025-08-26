from typing import List
import os
import glob

from eval_protocol.models import EvaluationRow, Message


def collect_dataset() -> List[EvaluationRow]:
    """
    Iterate through the dataset folder and create EvaluationRow objects.

    For each folder named "task_<n>", reads "task.txt" and "ground_truth.md"
    and creates an EvaluationRow where:
    - messages contains a user message with the task content
    - ground_truth contains the contents of ground_truth.md
    """
    dataset_rows = []
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

    # Find all task folders (task_<n>)
    task_folders = glob.glob(os.path.join(dataset_path, "task_*"))

    for task_folder in sorted(task_folders):
        task_name = os.path.basename(task_folder)

        # Read task.txt
        task_file = os.path.join(task_folder, "task.txt")
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r", encoding="utf-8") as f:
            task_content = f.read().strip()

        # Read ground_truth.md
        ground_truth_file = os.path.join(task_folder, "ground_truth.md")
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth_content = f.read().strip()

        # Create user message with the task
        user_message = Message(role="user", content=task_content)

        # Create EvaluationRow
        evaluation_row = EvaluationRow(messages=[user_message], ground_truth=ground_truth_content)

        dataset_rows.append(evaluation_row)

    return dataset_rows
