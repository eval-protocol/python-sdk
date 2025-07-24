import ast
import copy
import importlib
import inspect  # Import inspect for method signature
import json
import os
import sys

import pyarrow.feather as feather
import yaml
from datasets import load_from_disk

# Add the root directory and verifiers directory to the Python path
sys.path.append("/home/bchen/home/reward-kit")
sys.path.append("references/verifiers")

# Import BFCLSimAPIResource
from eval_protocol.agent_v2.resources.bfcl_sim_api_resource import BFCLSimAPIResource


# Helper function to parse function calls - Corrected to handle positional and keyword arguments
def _parse_function_call(func_call_str: str):
    """
    Parses a function call string into a JSON-like dictionary,
    preserving original keyword arguments and including positional arguments.

    :param func_call_str: String representation of a function call.
    :return: JSON-like dictionary with function name and arguments.
    """
    try:
        # Parse the function call string into an AST node
        tree = ast.parse(func_call_str, mode="eval")

        # Ensure it is a function call
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Input is not a valid function call.")

        # Extract function name
        func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None
        if not func_name:
            raise ValueError("Could not determine function name.")

        # Extract arguments
        args_dict = {}

        # Handle keyword arguments
        for kw in tree.body.keywords:
            args_dict[kw.arg] = ast.literal_eval(kw.value)

        # Handle positional arguments
        for i, arg in enumerate(tree.body.args):
            # Use a generic name for positional arguments
            args_dict[f"pos_arg_{i}"] = ast.literal_eval(arg)

        json_obj = {"name": func_name, "args": args_dict}

        return json_obj

    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Error parsing function call string '{func_call_str}': {e}")


# Define input and output paths
input_dataset_dir = "references/verifiers/verifiers/data/bfcl_dataset"
output_dir = "evaluations/bfcl/tasks"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset using the datasets library
dataset = load_from_disk(input_dataset_dir)

# Iterate through the dataset rows and create task definition files
for row in dataset:
    task_id = row["id"]
    question = row["question"]
    initial_config_str = row["initial_config"]
    involved_classes = row["involved_classes"]
    answer_str = row["answer"]
    original_prompt_messages = row["prompt"]  # This is List[Dict]
    user_question_bank_str = row["user_question_bank"]

    # Parse JSON strings
    initial_config = json.loads(initial_config_str) if initial_config_str else {}
    ground_truth_function_calls = json.loads(answer_str) if answer_str else []
    user_question_bank = json.loads(user_question_bank_str) if user_question_bank_str else []

    # Construct the messages list for the YAML: sequence of user turns only.
    # The original BFCL prompt_messages usually has [system_prompt, first_user_message_dict]
    # We only want the user messages.
    messages_for_yaml = []

    # Extract first user message if original_prompt_messages has at least two messages
    # and the second one is a user message.
    if isinstance(original_prompt_messages, list) and len(original_prompt_messages) > 1:
        first_user_message_candidate = original_prompt_messages[1]
        if isinstance(first_user_message_candidate, dict) and first_user_message_candidate.get("role") == "user":
            messages_for_yaml.append(first_user_message_candidate)
        else:
            # If the structure is different, log a warning and potentially add all non-system prompts
            print(
                f"Warning: Unexpected structure in original_prompt_messages for task {task_id}. Expected [system, user], got: {original_prompt_messages}"
            )
            for msg_dict in original_prompt_messages:
                if isinstance(msg_dict, dict) and msg_dict.get("role") == "user":
                    messages_for_yaml.append(msg_dict)
    elif (
        isinstance(original_prompt_messages, list)
        and len(original_prompt_messages) == 1
        and isinstance(original_prompt_messages[0], dict)
        and original_prompt_messages[0].get("role") == "user"
    ):
        # Handle cases where prompt_messages might only contain a single user message
        messages_for_yaml.append(original_prompt_messages[0])

    # Append subsequent user turns from user_question_bank
    # Each item in user_question_bank is a list representing a turn, which might contain multiple messages.
    # We need to ensure these are correctly added. The YAML expects a flat list of message dicts.
    for turn_message_list in user_question_bank:
        if isinstance(turn_message_list, list):
            for user_msg_dict in turn_message_list:
                if isinstance(user_msg_dict, dict) and user_msg_dict.get("role") == "user":
                    messages_for_yaml.append(user_msg_dict)
                else:
                    print(
                        f"Warning: Skipping non-user message or invalid format in user_question_bank for task {task_id}: {user_msg_dict}"
                    )
        elif (
            isinstance(turn_message_list, dict) and turn_message_list.get("role") == "user"
        ):  # If a turn is a single message dict
            messages_for_yaml.append(turn_message_list)
        else:
            print(f"Warning: Unexpected item format in user_question_bank for task {task_id}: {turn_message_list}")

    # --- Generate Ground Truth Final State ---
    gt_env_instances = {}
    for class_name in involved_classes:
        if class_name not in gt_env_instances:
            module_name = BFCLSimAPIResource.CLASS_FILE_PATH_MAPPING[class_name]
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            instance = class_()

            if class_name not in BFCLSimAPIResource.STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                instance._load_scenario(copy.deepcopy(class_initial_config))

            gt_env_instances[class_name] = instance

    # Now execute ground truth function calls on this gt_env_instances
    def execute_gt_calls(env_instances, gt_calls_per_turn):
        for gt_calls_turn_str in gt_calls_per_turn:
            for func_call_str in gt_calls_turn_str:
                try:
                    func_call = _parse_function_call(func_call_str)
                    tool_name = func_call["name"]
                    tool_args = func_call["args"]

                    found_method = False
                    for instance in env_instances.values():
                        if hasattr(instance, tool_name):
                            found_method = True
                            tool_func = getattr(instance, tool_name)
                            try:
                                # Execute the tool call
                                # Need to map generic pos_arg_X to actual parameter names
                                # This is complex. Let's try passing all args_dict and see if BFCL env methods handle it.
                                # If not, we might need to inspect method signatures here.
                                # Based on the previous error, it seems they expect specific keyword args or positional.
                                # Let's revert to the previous approach of only extracting keyword args,
                                # and assume positional args in ground truth are a different format or not intended.
                                # Reverting _parse_function_call to only handle keyword args as before.
                                # The issue might be in the ground truth data format itself or how BFCL envs handle args.

                                # Let's try a different approach: inspect the method signature and map positional args.
                                sig = inspect.signature(tool_func)
                                bound_args = sig.bind(**tool_args)  # Try binding with extracted args
                                bound_args.apply_defaults()  # Apply default values

                                # Execute the tool call with bound arguments
                                tool_func(*bound_args.args, **bound_args.kwargs)

                            except TypeError as e:
                                print(f"TypeError executing ground truth tool {tool_name} with args {tool_args}: {e}")
                                pass
                            except Exception as e:
                                print(f"Error executing ground truth tool {tool_name} with args {tool_args}: {e}")
                                pass
                            break
                    if not found_method:
                        print(f"Ground truth tool {tool_name} not found in env instances.")

                except ValueError as e:  # Catch errors from _parse_function_call
                    print(f"Parsing error for ground truth call '{func_call_str}': {e}")
                    pass
                except Exception as e:
                    print(f"Unexpected error during ground truth call processing '{func_call_str}': {e}")
                    pass

    execute_gt_calls(gt_env_instances, ground_truth_function_calls)

    # Get the comparable state of the ground truth resource
    temp_gt_resource = BFCLSimAPIResource(env_instances=gt_env_instances)
    ground_truth_comparable_state = temp_gt_resource.get_comparable_state()
    # --- End Generate Ground Truth Final State ---

    # Construct the v2 task definition dictionary
    task_definition = {
        "name": task_id,
        "description": f"BFCL task: {question}",
        "resource_type": "BFCLSimAPIResource",
        "base_resource_config": {
            "involved_classes": involved_classes,
            "initial_config": initial_config,
        },
        "evaluation_criteria": {
            # Populate the new explicit fields
            "ground_truth_function_calls": ground_truth_function_calls,
            "ground_truth_comparable_state": ground_truth_comparable_state,
        },
        "messages": messages_for_yaml,  # Use the cleaned messages list
        "reward_function_path": "eval_protocol.rewards.bfcl_reward",
        # Add poc_max_turns based on the number of user turns, or a default
        # This ensures the orchestrator processes all defined user turns if poc_max_turns is not explicitly set lower.
        # "poc_max_turns": len(messages_for_yaml) # Or a fixed default like 10 if preferred
    }

    # Define the output file path
    output_file = os.path.join(output_dir, f"{task_id}.yaml")

    # Write the task definition to a YAML file
    with open(output_file, "w") as f:
        yaml.dump(task_definition, f, indent=2)

    # print(f"Created task definition: {output_file}") # Suppress successful creation messages for clarity

print("Dataset conversion complete (errors during ground truth execution logged above).")
