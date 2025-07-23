import json  # Moved import to top of file
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.typed_interface import reward_function


@reward_function
def evaluate(
    messages: List[Message],
    ground_truth: str,
    final_filesystem_state: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Pure evaluation function that compares actual vs expected filesystem state.
    This function is designed to be easily used by the Reward Kit framework for multiple rollouts.

    Args:
        messages: List of conversation messages from the rollout.
        ground_truth: Expected outcome description for this task.
        final_filesystem_state: The state of the filesystem at the end of the rollout,
                                 provided by the framework via the capture_function.
        task_description: The description of the task being evaluated.

    Returns:
        EvaluateResult containing the score and detailed metrics.
    """
    # Extract the model's final response
    completion = messages[-1].content if messages and messages[-1].content else ""

    # Use provided arguments directly
    # final_filesystem_state is the raw output from the directory_tree tool
    # which is like: {"content": [{"type": "text", "text": "[...json array...]"}]}
    # We need to parse the JSON string within it.

    actual_root_items = []
    raw_state_capture = final_filesystem_state

    if (
        isinstance(raw_state_capture, dict)
        and raw_state_capture.get("content")
        and isinstance(raw_state_capture["content"], list)
        and len(raw_state_capture["content"]) > 0
        and isinstance(raw_state_capture["content"][0], dict)
        and raw_state_capture["content"][0].get("type") == "text"
        and isinstance(raw_state_capture["content"][0].get("text"), str)
    ):
        try:
            # The 'text' field contains the JSON string of the directory tree
            parsed_fs_tree = json.loads(raw_state_capture["content"][0]["text"])
            if isinstance(parsed_fs_tree, list):  # Ensure it's a list after parsing
                actual_root_items = parsed_fs_tree
            else:
                print(
                    f"Warning: Parsed directory_tree output is not a list: {type(parsed_fs_tree)}"
                )
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from directory_tree output: {e}")
            actual_root_items = []  # Default to empty list on error
    elif isinstance(
        raw_state_capture, list
    ):  # If it's already a parsed list (e.g. from older direct passing)
        actual_root_items = raw_state_capture
    else:
        print(
            f"Warning: final_filesystem_state is not in expected format. Type: {type(raw_state_capture)}"
        )

    current_task_description = task_description if task_description is not None else ""

    # Parse ground truth expectation
    expected_state = parse_ground_truth(ground_truth)

    # Compare actual vs expected filesystem state
    filesystem_score = calculate_filesystem_match_score(
        actual_root_items, expected_state
    )

    # Analyze the conversation for task completion indicators
    completion_score = analyze_task_completion(completion, current_task_description)

    # Combined score
    # For now, let's simplify: if filesystem match is perfect, score is 1, otherwise it's completion_score * fs_score
    # This gives more weight to actual filesystem changes.
    if filesystem_score == 1.0:
        final_score = 1.0  # Prioritize perfect filesystem match
    else:
        # If fs match is not perfect, weigh it with completion.
        # If list operation, completion score is less critical than actual list content.
        # Let's make fs_score dominant for now if it's not 0.
        if (
            expected_state.get("expected_operations") == ["list"]
            and filesystem_score > 0
        ):
            final_score = filesystem_score  # For list, fs_score is primary
        else:
            final_score = min(filesystem_score * 0.7 + completion_score * 0.3, 1.0)

    # Add logging for scores
    print(
        f"[INFO] Sample ID from reward_function: (not available here, but for task: {current_task_description or ground_truth})"
    )
    print(f"[INFO] Filesystem Score: {filesystem_score:.2f}")
    print(f"[INFO] Completion Score: {completion_score:.2f}")
    print(f"[INFO] Final Score: {final_score:.2f}")

    return EvaluateResult(
        score=final_score,
        is_score_valid=True,
        reason=f"Filesystem match: {filesystem_score:.2f}, Task completion: {completion_score:.2f}",
        metrics={
            "filesystem_accuracy": MetricResult(
                score=filesystem_score,
                is_score_valid=True,
                reason="Comparison of expected vs actual filesystem state",
            ),
            "task_completion": MetricResult(
                score=completion_score,
                is_score_valid=True,
                reason="Analysis of completion indicators in model response",
            ),
        },
    )


def parse_ground_truth(ground_truth: str) -> Dict[str, Any]:
    """Parse ground truth string into expected filesystem state."""
    if "move file_to_move.txt to target_dir" in ground_truth:
        return {
            "expected_operations": ["move"],
            "source_files": ["file_to_move.txt"],
            "target_location": "target_dir/file_to_move.txt",
        }
    elif "list contents of /data/source_dir" in ground_truth:
        return {
            "expected_operations": ["list"],
            "path_to_list": "/data/source_dir",
            "expected_files": ["file_to_move.txt", "sample.txt"],
        }
    elif "create report.txt with specified content" in ground_truth:
        return {
            "expected_operations": ["create"],
            "target_file": "report.txt",
            "expected_content": "Task completed successfully",
        }
    return {"expected_operations": []}


def calculate_filesystem_match_score(
    actual_root_items: List[Dict], expected_state: Dict
) -> float:
    """
    Calculate how well the actual filesystem state matches expectations.
    actual_root_items is a list of dicts representing items in /data.
    """
    if not expected_state.get("expected_operations"):
        return 1.0  # No specific operation to check, assume success if no error by this point.

    operations = expected_state.get("expected_operations", [])

    if "move" in operations:
        return check_move_operation(actual_root_items, expected_state)
    elif "list" in operations:
        return check_list_operation(actual_root_items, expected_state)
    elif "create" in operations:
        return check_create_operation(actual_root_items, expected_state)

    return 0.0


def find_node_by_path(
    root_items: List[Dict], relative_path_parts: List[str]
) -> Optional[Dict]:
    """Helper to find a node (file or dir) by its relative path parts from root_items."""
    current_level_items = root_items
    node = None
    for i, part_name in enumerate(relative_path_parts):
        found_part_node = None
        for item in current_level_items:
            if item.get("name") == part_name:
                # If it's the last part, it can be a file or dir.
                # If not last part, must be a dir to continue.
                if i == len(relative_path_parts) - 1:  # Last part
                    found_part_node = item
                    break
                elif item.get("type") == "directory":  # Intermediate part must be dir
                    found_part_node = item
                    break

        if found_part_node:
            node = found_part_node
            if node.get("type") == "directory" and i < len(relative_path_parts) - 1:
                current_level_items = node.get("children", [])
            elif i < len(relative_path_parts) - 1:  # Intermediate part is not a dir
                return None
        else:
            return None  # Path part not found
    return node


def check_move_operation(actual_root_items: List[Dict], expected_state: Dict) -> float:
    """Check if file was moved correctly. target_location is relative to /data."""
    target_location_rel = expected_state.get(
        "target_location", ""
    )  # e.g., "target_dir/file_to_move.txt"

    target_path_parts = [part for part in target_location_rel.split("/") if part]

    if not target_path_parts:
        return 0.0

    moved_file_node = find_node_by_path(actual_root_items, target_path_parts)

    if moved_file_node and moved_file_node.get("type") == "file":
        # Optionally, also check if the file is GONE from the source.
        # This requires knowing the original source path.
        # For now, just checking existence at target.
        return 1.0
    return 0.0


def check_list_operation(actual_root_items: List[Dict], expected_state: Dict) -> float:
    """Check if directory listing is correct. path_to_list is absolute from container root."""
    path_to_list_abs = expected_state.get(
        "path_to_list", ""
    )  # e.g., "/data/source_dir"

    # Convert absolute path to list of parts relative to /data
    # e.g., "/data/source_dir" -> ["source_dir"]
    # e.g., "/data" -> [] (listing root of /data)
    relative_path_parts = [
        part for part in path_to_list_abs.split("/") if part and part != "data"
    ]

    listed_dir_node = None
    if not relative_path_parts:
        # This case means we are listing the root of what was captured (e.g. /data itself)
        # So, the children are actual_root_items directly.
        # This path is not taken by current dataset.jsonl which has /data/source_dir
        # If it were /data, actual_files would be [item.get("name") for item in actual_root_items if item.get("type") == "file"]
        # For now, if this path is hit, it's an unexpected configuration for this function's design.
        # However, if path_to_list is "/data", then actual_root_items are its children.
        # This logic could be expanded if listing the root of the capture path is a valid scenario.
        # For this example, we assume path_to_list targets a named directory within the capture_path.
        # If path_to_list was intended to be the root of capture, expected_files would be checked against actual_root_items.
        # Given current dataset, this branch implies an issue or a need for more robust path handling for root.
        return 0.0
    else:
        listed_dir_node = find_node_by_path(actual_root_items, relative_path_parts)

    if not listed_dir_node or listed_dir_node.get("type") != "directory":
        return 0.0

    actual_files = [
        child.get("name")
        for child in listed_dir_node.get("children", [])
        if child.get("type") == "file"
    ]
    expected_files = expected_state.get("expected_files", [])

    if (
        not expected_files
    ):  # If ground truth doesn't specify files, any successful list is 1.0
        return 1.0

    match_count = 0
    for ef in expected_files:
        if ef in actual_files:
            match_count += 1

    score = float(match_count) / len(expected_files) if expected_files else 1.0
    return score


def check_create_operation(
    actual_root_items: List[Dict], expected_state: Dict
) -> float:
    """Check if file was created. target_file is relative to /data."""
    target_filename_rel = expected_state.get("target_file", "")  # e.g., "report.txt"
    # expected_content = expected_state.get("expected_content", "") # Content check not possible with directory_tree

    target_path_parts = [part for part in target_filename_rel.split("/") if part]

    if not target_path_parts:
        return 0.0

    created_file_node = find_node_by_path(actual_root_items, target_path_parts)

    if created_file_node and created_file_node.get("type") == "file":
        # Content check is not feasible with directory_tree output as it doesn't include content.
        # If content check was required, reward function would need to use 'read_file' tool.
        return 1.0  # File exists

    return 0.0


def analyze_task_completion(completion: str, task_description: str) -> float:
    """Analyze the model's response for task completion indicators."""
    if not completion:
        return 0.0

    completion_lower = completion.lower()

    # Positive indicators
    positive_indicators = [
        "successfully moved",
        "file has been moved",
        "completed the task",
        "operation successful",
        "task completed",
        "successfully copied",
        "file created",
        "successfully created",
    ]

    # Negative indicators
    negative_indicators = [
        "error",
        "failed",
        "cannot",
        "unable",
        "not found",
        "permission denied",
        "access denied",
    ]

    score = 0.5  # Base score

    # Boost for positive indicators
    if any(indicator in completion_lower for indicator in positive_indicators):
        score += 0.3

    # Penalty for negative indicators
    if any(indicator in completion_lower for indicator in negative_indicators):
        score -= 0.4

    # Bonus for detailed responses
    if len(completion) > 50:
        score += 0.2

    return max(0.0, min(1.0, score))
