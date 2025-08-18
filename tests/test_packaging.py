import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_protocol.packaging import (
    DEFAULT_PYTHON_VERSION,
    _resolve_module_path_and_name,
    generate_dockerfile_content,
)

# Ensure the dummy_rewards module can be found if tests are run from root
# and CWD is root, by adding current dir to sys.path if not already there.
# This helps _resolve_module_path_and_name find modules in CWD.
if Path.cwd().as_posix() not in sys.path:
    sys.path.insert(0, Path.cwd().as_posix())


class TestPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a unique dummy reward function file for testing to avoid race conditions
        import uuid

        unique_id = str(uuid.uuid4()).replace("-", "")[:8]
        cls.DUMMY_REWARD_MODULE_NAME = f"dummy_test_reward_module_for_packaging_{unique_id}"
        cls.DUMMY_REWARD_MODULE_FILENAME = f"{cls.DUMMY_REWARD_MODULE_NAME}.py"
        cls.DUMMY_REWARD_FUNCTION_NAME = "my_dummy_reward_func"
        cls.DUMMY_FUNCTION_REF = f"{cls.DUMMY_REWARD_MODULE_NAME}.{cls.DUMMY_REWARD_FUNCTION_NAME}"

        cls.DUMMY_REWARD_MODULE_CONTENT = f"""
from eval_protocol.typed_interface import reward_function

@reward_function(id="test-dummy-packaging", requirements="requests==2.25.1\\nnumpy>=1.20.0")
def {cls.DUMMY_REWARD_FUNCTION_NAME}(messages, ground_truth=None):
    return {{"score": 1.0, "reason": "Dummy success"}}
"""

        with open(cls.DUMMY_REWARD_MODULE_FILENAME, "w") as f:
            f.write(cls.DUMMY_REWARD_MODULE_CONTENT)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.DUMMY_REWARD_MODULE_FILENAME):
            os.remove(cls.DUMMY_REWARD_MODULE_FILENAME)

    def test_resolve_module_path_and_name_simple_module(self):
        """Test resolving a simple .py module in CWD."""
        # Create a temporary dummy module file
        temp_module_name = "temp_resolve_test_module"
        temp_module_filename = f"{temp_module_name}.py"
        with open(temp_module_filename, "w") as f:
            f.write("def test_func(): pass\n")

        try:
            resolved = _resolve_module_path_and_name(f"{temp_module_name}.test_func")
            self.assertIsNotNone(resolved)
            path_to_copy, top_level_name, copy_dest_name = resolved
            self.assertEqual(path_to_copy, Path(os.getcwd()) / temp_module_filename)
            self.assertEqual(top_level_name, temp_module_name)
            self.assertEqual(copy_dest_name, temp_module_filename)
        finally:
            if os.path.exists(temp_module_filename):
                os.remove(temp_module_filename)

    def test_generate_dockerfile_with_inline_requirements(self):
        requirements_content = "flask==2.0.1\npydantic<2.0"
        dockerfile = generate_dockerfile_content(
            function_ref=self.DUMMY_FUNCTION_REF,
            inline_requirements_content=requirements_content,
        )
        self.assertIsNotNone(dockerfile)
        self.assertIn(f"FROM python:{DEFAULT_PYTHON_VERSION}-slim", dockerfile)
        self.assertIn("COPY . .", dockerfile)
        self.assertIn("RUN pip install --no-cache-dir .", dockerfile)  # Installs eval_protocol

        # Check for inline requirements installation
        # Need to handle potential shell escaping in the echo command
        escaped_requirements = requirements_content.replace("\\", "\\\\").replace("'", "'\\''")
        self.assertIn(
            f"RUN echo '{escaped_requirements}' > /app/generated_requirements.txt",
            dockerfile,
        )
        self.assertIn(
            "RUN pip install --no-cache-dir -r /app/generated_requirements.txt",
            dockerfile,
        )

        self.assertIn(
            f"CMD python -m eval_protocol.generic_server {self.DUMMY_FUNCTION_REF}",
            dockerfile,
        )

    def test_generate_dockerfile_with_user_requirements_path(self):
        user_req_path = "my_custom_requirements.txt"
        dockerfile = generate_dockerfile_content(
            function_ref=self.DUMMY_FUNCTION_REF, user_requirements_path=user_req_path
        )
        self.assertIsNotNone(dockerfile)
        self.assertIn(
            f"RUN if [ -f {user_req_path} ]; then pip install --no-cache-dir -r {user_req_path};",
            dockerfile,
        )

    def test_generate_dockerfile_no_extra_requirements(self):
        dockerfile = generate_dockerfile_content(function_ref=self.DUMMY_FUNCTION_REF)
        self.assertIsNotNone(dockerfile)
        self.assertNotIn("generated_requirements.txt", dockerfile)
        self.assertNotIn("-r my_custom_requirements.txt", dockerfile)  # Assuming this is specific enough

    def test_generate_dockerfile_unresolvable_function_ref(self):
        dockerfile = generate_dockerfile_content(function_ref="non_existent_module.bad_func")
        self.assertIsNone(dockerfile)


if __name__ == "__main__":
    unittest.main()
