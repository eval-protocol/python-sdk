"""
Utility for dynamically loading modules and functions.
"""

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def load_function(import_path: str) -> Callable[..., Any]:
    """
    Dynamically loads a function given its full import path.
    Example: "my_package.my_module.my_function"
    """
    try:
        module_path, function_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        if not callable(func):
            raise AttributeError(f"'{function_name}' in module '{module_path}' is not callable.")
        logger.info(f"Successfully loaded function '{function_name}' from '{module_path}'.")
        return func
    except ImportError as e:
        logger.error(f"Failed to import module from path '{import_path}': {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find or access function in path '{import_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading function from '{import_path}': {e}")
        raise


# Example usage:
# if __name__ == '__main__':
#     try:
#         # Assuming you have a eval_protocol.rewards.math module with math_reward function
#         math_reward_func = load_function("eval_protocol.rewards.math.math_reward")
#         print(f"Loaded: {math_reward_func}")
#         # You could then call it, e.g., if it took simple args: math_reward_func(arg1="test")
#     except Exception as e:
#         print(f"Test loading failed: {e}")

#     try:
#         # Test with a non-existent function
#         load_function("eval_protocol.rewards.math.non_existent_function")
#     except Exception as e:
#         print(f"Test loading non-existent function failed as expected: {e}")

#     try:
#         # Test with a non-existent module
#         load_function("non_existent_package.non_existent_module.some_function")
#     except Exception as e:
#         print(f"Test loading non-existent module failed as expected: {e}")
