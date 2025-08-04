import os
from unittest.mock import mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from eval_protocol import config as rk_config  # Alias to avoid conflict
from eval_protocol.config import (
    CONFIG_FILE_NAME,
    AWSLambdaConfig,
    GCPCloudRunConfig,
    RewardKitConfig,
)


# Helper to reset config state for isolation if module-level cache is an issue
@pytest.fixture(autouse=True)
def reset_config_module_state():
    rk_config._loaded_config = None
    rk_config._config_file_path = None
    yield
    rk_config._loaded_config = None
    rk_config._config_file_path = None


class TestFindConfigFile:
    def test_find_in_current_dir(self, tmp_path):
        """Test finding config in the current directory."""
        (tmp_path / CONFIG_FILE_NAME).write_text("test: content")
        found_path = rk_config.find_config_file(start_path=str(tmp_path))
        assert found_path == str(tmp_path / CONFIG_FILE_NAME)

    def test_find_in_parent_dir(self, tmp_path):
        """Test finding config in a parent directory."""
        parent_dir = tmp_path
        child_dir = tmp_path / "subdir"
        child_dir.mkdir()
        (parent_dir / CONFIG_FILE_NAME).write_text("test: content")

        found_path = rk_config.find_config_file(start_path=str(child_dir))
        assert found_path == str(parent_dir / CONFIG_FILE_NAME)

    def test_not_found(self, tmp_path):
        """Test when no config file is found up to root."""
        # To ensure find_config_file doesn't find any actual config file during this test,
        # we mock os.path.isfile to always return False.
        # The start_path (tmp_path) ensures it starts in an isolated directory.
        with patch("os.path.isfile", return_value=False):
            found_path = rk_config.find_config_file(start_path=str(tmp_path))
            assert found_path is None


class TestLoadConfig:
    def test_load_no_config_file_found(self):
        """Test loading config when no rewardkit.yaml exists."""
        with patch("eval_protocol.config.find_config_file", return_value=None):
            cfg = rk_config.get_config()
            assert isinstance(cfg, RewardKitConfig)
            # Check for default values
            assert cfg.default_deployment_target == "fireworks"
            assert cfg.gcp_cloud_run is not None
            assert cfg.gcp_cloud_run.project_id is None
            assert cfg.aws_lambda is not None
            assert cfg.evaluator_endpoint_keys == {}

    def test_load_empty_config_file(self, tmp_path):
        """Test loading an empty rewardkit.yaml file."""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text("")  # Empty file

        cfg = rk_config.load_config(str(config_file))
        assert isinstance(cfg, RewardKitConfig)
        assert cfg.default_deployment_target == "fireworks"  # Should use Pydantic defaults

    def test_load_valid_full_config_file(self, tmp_path):
        """Test loading a valid config file with all sections."""
        content = """
default_deployment_target: gcp-cloud-run
gcp_cloud_run:
  project_id: "my-gcp-proj"
  region: "us-west1"
  service_name_template: "eval-{evaluator_id}"
  default_auth_mode: "iam"
  secrets:
    API_KEY: "gcp_secret_id_for_api_key"
aws_lambda:
  region: "eu-central-1"
  function_name_template: "lambda-eval-{evaluator_id}"
  default_auth_mode: "mtls-client-auth"
  secrets:
    OTHER_KEY: "aws_secret_arn_for_other_key"
evaluator_endpoint_keys:
  my_eval_1: "key123"
"""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text(content)

        cfg = rk_config.load_config(str(config_file))
        assert cfg.default_deployment_target == "gcp-cloud-run"
        assert cfg.gcp_cloud_run.project_id == "my-gcp-proj"
        assert cfg.gcp_cloud_run.region == "us-west1"
        assert cfg.gcp_cloud_run.service_name_template == "eval-{evaluator_id}"
        assert cfg.gcp_cloud_run.default_auth_mode == "iam"
        assert cfg.gcp_cloud_run.secrets == {"API_KEY": "gcp_secret_id_for_api_key"}
        assert cfg.aws_lambda.region == "eu-central-1"
        assert cfg.aws_lambda.function_name_template == "lambda-eval-{evaluator_id}"
        assert cfg.aws_lambda.default_auth_mode == "mtls-client-auth"
        assert cfg.aws_lambda.secrets == {"OTHER_KEY": "aws_secret_arn_for_other_key"}
        assert cfg.evaluator_endpoint_keys == {"my_eval_1": "key123"}

    def test_load_valid_partial_config_file(self, tmp_path):
        """Test loading a valid config with some optional sections/fields missing."""
        content = """
default_deployment_target: local
gcp_cloud_run:
  project_id: "my-gcp-proj"
# aws_lambda is missing, should use default
# evaluator_endpoint_keys is missing, should use default
"""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text(content)

        cfg = rk_config.load_config(str(config_file))
        assert cfg.default_deployment_target == "local"
        assert cfg.gcp_cloud_run.project_id == "my-gcp-proj"
        assert cfg.gcp_cloud_run.region is None  # Default for optional field
        assert cfg.gcp_cloud_run.default_auth_mode == "api-key"  # Default from model
        assert cfg.aws_lambda is not None  # Default AWSLambdaConfig object
        assert cfg.aws_lambda.region is None
        assert cfg.evaluator_endpoint_keys == {}  # Default empty dict

    def test_load_malformed_yaml(self, tmp_path, capsys):
        """Test loading a file with invalid YAML syntax."""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text("default_deployment_target: gcp-cloud-run\n  bad_indent: true")

        cfg = rk_config.load_config(str(config_file))
        captured = capsys.readouterr()
        assert "Error parsing YAML" in captured.out
        assert isinstance(cfg, RewardKitConfig)  # Should return default
        assert cfg.default_deployment_target == "fireworks"

    def test_load_schema_violation(self, tmp_path, capsys):
        """Test loading valid YAML but with schema violations (wrong types)."""
        content = """
default_deployment_target: "invalid_target_option" # Not in Literal
gcp_cloud_run:
  project_id: 12345 # Should be string
"""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text(content)

        cfg = rk_config.load_config(str(config_file))
        captured = capsys.readouterr()
        assert "Error validating configuration" in captured.out
        assert isinstance(cfg, RewardKitConfig)  # Should return default
        assert cfg.default_deployment_target == "fireworks"

    def test_config_caching(self, tmp_path):
        """Test that get_config() returns a cached instance."""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text("default_deployment_target: local")

        # Patch find_config_file to control its result for this test
        with patch("eval_protocol.config.find_config_file", return_value=str(config_file)) as mock_find:
            cfg1 = rk_config.get_config()
            assert cfg1.default_deployment_target == "local"
            mock_find.assert_called_once()  # find_config_file should be called

            # Call get_config() again, should use cache
            cfg2 = rk_config.get_config()
            assert cfg2 is cfg1  # Should be the same instance
            mock_find.assert_called_once()  # find_config_file should NOT be called again

            # Force reload by providing path to load_config
            rk_config.load_config(str(config_file))  # This will reload
            # If find_config_file was patched at module level, this call to load_config
            # would use the patched version. The fixture resets _loaded_config.
            # The number of calls to find_config_file depends on how load_config is structured.
            # The current load_config structure: if path is given, it uses it. If not, it calls find_config_file.
            # So, load_config(path) doesn't call find_config_file.

            # Reset and call get_config again after a direct load_config(path)
            rk_config._loaded_config = None
            rk_config._config_file_path = None  # Ensure it's not using the path cache

            cfg3 = rk_config.get_config()
            assert cfg3.default_deployment_target == "local"
            # mock_find should now have been called twice (once for cfg1, once for cfg3)
            assert mock_find.call_count == 2

    def test_load_specific_path_overrides_cache_or_find(self, tmp_path):
        """Test that load_config(specific_path) loads that file."""
        default_config_file = tmp_path / "default_loc" / CONFIG_FILE_NAME
        default_config_file.parent.mkdir()
        default_config_file.write_text("default_deployment_target: fireworks")

        specific_config_file = tmp_path / "specific_loc" / CONFIG_FILE_NAME
        specific_config_file.parent.mkdir()
        specific_config_file.write_text("default_deployment_target: gcp-cloud-run")

        # Patch find_config_file to return the "default" location
        with patch(
            "eval_protocol.config.find_config_file",
            return_value=str(default_config_file),
        ):
            # First, get_config might load the one found by find_config_file
            cfg_found = rk_config.get_config()
            assert cfg_found.default_deployment_target == "fireworks"

            # Now, load a specific one
            cfg_specific = rk_config.load_config(str(specific_config_file))
            assert cfg_specific.default_deployment_target == "gcp-cloud-run"

            # get_config() should now return the specifically loaded one
            cfg_after_specific_load = rk_config.get_config()
            assert cfg_after_specific_load is cfg_specific
            assert cfg_after_specific_load.default_deployment_target == "gcp-cloud-run"
