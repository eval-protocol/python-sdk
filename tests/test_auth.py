import configparser  # Import the original for type hinting if needed, but not for spec.
import os

# Import the original ConfigParser for use in spec if absolutely necessary,
# though direct configuration of the mock instance is preferred.
from configparser import ConfigParser as OriginalConfigParser
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import the SUT
from eval_protocol.auth import (
    AUTH_INI_FILE,
    get_fireworks_account_id,
    get_fireworks_api_key,
)

# Test data
TEST_ENV_API_KEY = "test_env_api_key_123"
TEST_ENV_ACCOUNT_ID = "test_env_account_id_456"
INI_API_KEY = "ini_api_key_abc"
INI_ACCOUNT_ID = "ini_account_id_def"


@pytest.fixture(autouse=True)
def clear_env_vars_fixture():
    env_vars_to_clear = ["FIREWORKS_API_KEY", "FIREWORKS_ACCOUNT_ID"]
    original_values = {var: os.environ.get(var) for var in env_vars_to_clear}
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    yield
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


# --- Tests for get_fireworks_api_key ---


def test_get_api_key_from_env():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY
    assert get_fireworks_api_key() == TEST_ENV_API_KEY


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")  # Mocks the ConfigParser class
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Ensure simple parse finds nothing
def test_get_api_key_from_ini(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    # Configure the instance that configparser.ConfigParser() will return
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    # Simulate key found in [fireworks] section
    def has_option_fireworks_true(section, option):
        if section == "fireworks":
            return option == "api_key"
        return False  # Not in default or other sections for this test

    def get_fireworks_value(section, option):
        if section == "fireworks" and option == "api_key":
            return INI_API_KEY
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_fireworks_true
    mock_parser_instance.get.side_effect = get_fireworks_value
    # Ensure 'fireworks' section itself exists
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"  # For "fireworks" in config check

    with patch(
        "builtins.open", mock_open(read_data="[fireworks]\napi_key = foo")
    ):  # Actual read_data not used by mock parser
        assert get_fireworks_api_key() == INI_API_KEY

    mock_path_exists.assert_called_once_with()
    mock_ConfigParser_class.assert_called_once_with()  # Class was instantiated
    mock_parser_instance.read.assert_called_once_with(AUTH_INI_FILE)
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


def test_get_api_key_env_overrides_ini():
    os.environ["FIREWORKS_API_KEY"] = TEST_ENV_API_KEY
    with (
        patch("pathlib.Path.exists") as mock_path_exists,
        patch("configparser.ConfigParser") as mock_ConfigParser_class,
        patch("eval_protocol.auth._parse_simple_auth_file") as mock_parse_simple,
    ):
        assert get_fireworks_api_key() == TEST_ENV_API_KEY
        mock_parse_simple.assert_not_called()  # Env var should be checked first
        mock_path_exists.assert_not_called()
        mock_ConfigParser_class.assert_not_called()


@patch("pathlib.Path.exists", return_value=False)
def test_get_api_key_not_found(mock_path_exists):
    # Ensure _parse_simple_auth_file is also considered if Path.exists is True but file is empty/no key
    with patch("eval_protocol.auth._parse_simple_auth_file", return_value={}) as mock_parse_simple:
        assert get_fireworks_api_key() is None
        # _get_credential_from_config_file checks AUTH_INI_FILE.exists() first
        # if it's false, _parse_simple_auth_file won't be called by it.
        # However, get_fireworks_api_key calls _get_credential_from_config_file,
        # which itself calls _parse_simple_auth_file if AUTH_INI_FILE.exists().
        # This test specifically tests when AUTH_INI_FILE does *not* exist.
        mock_parse_simple.assert_not_called()  # Because AUTH_INI_FILE.exists() is False
    mock_path_exists.assert_called_once_with()


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Simple parse finds nothing
def test_get_api_key_ini_exists_no_section(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    # Simulate MissingSectionHeaderError to trigger fallback within configparser logic (though simple parse is first)
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError("file", 1, "line")

    with patch(
        "builtins.open",  # This mock_open is for the configparser's attempt if it were reached
        mock_open(read_data="other_key = some_val_but_no_section_header\nanother=val"),
    ):
        assert get_fireworks_api_key() is None
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_api_key_ini_exists_no_key_option(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = lambda section, option: False

    with patch("builtins.open", mock_open(read_data="[fireworks]\nsome_other_key=foo")):
        assert get_fireworks_api_key() is None
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_api_key_ini_empty_value(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = (
        lambda section, option: section == "fireworks" and option == "api_key"
    )
    mock_parser_instance.get.side_effect = lambda section, option: (
        "" if section == "fireworks" and option == "api_key" else configparser.NoOptionError(option, section)
    )

    with patch("builtins.open", mock_open(read_data="[fireworks]\napi_key=")):
        assert get_fireworks_api_key() is None
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_api_key_from_ini_default_section_success(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    def has_option_logic(section, option):
        if section == "fireworks":
            return False
        return section == mock_parser_instance.default_section and option == "api_key"

    def get_logic(section, option):
        if section == mock_parser_instance.default_section and option == "api_key":
            return INI_API_KEY
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_logic
    mock_parser_instance.get.side_effect = get_logic
    mock_parser_instance.__contains__.side_effect = lambda item: item != "fireworks"

    with patch("builtins.open", mock_open(read_data="api_key = ini_api_key_abc")):
        assert get_fireworks_api_key() == INI_API_KEY
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
# We don't mock ConfigParser here because we are testing the _parse_simple_auth_file path directly
def test_get_api_key_from_ini_simple_parsing_success(mock_path_exists):
    file_content = f"api_key = {INI_API_KEY}\nother_key = value"
    # Patch open specifically for _parse_simple_auth_file if it's different from configparser's use
    with patch("eval_protocol.auth.open", mock_open(read_data=file_content), create=True):
        # Ensure configparser path is not taken or also returns None for api_key
        with patch("configparser.ConfigParser") as mock_ConfigParser_class_inner:
            mock_parser_instance = mock_ConfigParser_class_inner.return_value
            mock_parser_instance.read.return_value = []  # Simulate configparser finds nothing
            mock_parser_instance.has_option.return_value = False
            assert get_fireworks_api_key() == INI_API_KEY


# --- Tests for get_fireworks_account_id ---


def test_get_account_id_from_env():
    os.environ["FIREWORKS_ACCOUNT_ID"] = TEST_ENV_ACCOUNT_ID
    assert get_fireworks_account_id() == TEST_ENV_ACCOUNT_ID


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Ensure simple parse finds nothing
def test_get_account_id_from_ini(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    def has_option_fireworks_true(section, option):
        if section == "fireworks":
            return option == "account_id"
        return False

    def get_fireworks_value(section, option):
        if section == "fireworks" and option == "account_id":
            return INI_ACCOUNT_ID
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_fireworks_true
    mock_parser_instance.get.side_effect = get_fireworks_value
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"

    with patch("builtins.open", mock_open(read_data="[fireworks]\naccount_id = foo")):
        assert get_fireworks_account_id() == INI_ACCOUNT_ID

    mock_path_exists.assert_called_once_with()
    mock_ConfigParser_class.assert_called_once_with()
    mock_parser_instance.read.assert_called_once_with(AUTH_INI_FILE)
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


def test_get_account_id_env_overrides_ini():
    os.environ["FIREWORKS_ACCOUNT_ID"] = TEST_ENV_ACCOUNT_ID
    with (
        patch("pathlib.Path.exists") as mock_path_exists,
        patch("configparser.ConfigParser") as mock_ConfigParser_class,
        patch("eval_protocol.auth._parse_simple_auth_file") as mock_parse_simple,
    ):
        assert get_fireworks_account_id() == TEST_ENV_ACCOUNT_ID
        mock_parse_simple.assert_not_called()
        mock_path_exists.assert_not_called()
        mock_ConfigParser_class.assert_not_called()


@patch("pathlib.Path.exists", return_value=False)
def test_get_account_id_not_found(mock_path_exists):
    with patch("eval_protocol.auth._parse_simple_auth_file", return_value={}) as mock_parse_simple:
        assert get_fireworks_account_id() is None
        mock_parse_simple.assert_not_called()
    # With verify fallback using get_fireworks_api_key, exists() may be checked more than once
    assert mock_path_exists.call_count >= 1


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_account_id_ini_exists_no_section(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = configparser.MissingSectionHeaderError("file", 1, "line")
    with patch(
        "builtins.open",
        mock_open(read_data="other_key = some_val_but_no_section_header\nanother=val"),
    ):
        assert get_fireworks_account_id() is None
    # Fallback verify path may trigger a second simple parse for api_key; ensure at least one call
    assert mock_parse_simple.call_count >= 1


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_account_id_ini_exists_no_id_option(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = lambda section, option: False

    with patch("builtins.open", mock_open(read_data="[fireworks]\nsome_other_key=foo")):
        assert get_fireworks_account_id() is None
    # Fallback verify path may trigger a second simple parse for api_key; ensure at least one call
    assert mock_parse_simple.call_count >= 1


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_account_id_ini_empty_value(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]
    mock_parser_instance.__contains__.side_effect = lambda item: item == "fireworks"
    mock_parser_instance.has_option.side_effect = (
        lambda section, option: section == "fireworks" and option == "account_id"
    )
    mock_parser_instance.get.side_effect = lambda section, option: (
        "" if section == "fireworks" and option == "account_id" else configparser.NoOptionError(option, section)
    )
    with patch("builtins.open", mock_open(read_data="[fireworks]\naccount_id=")):
        assert get_fireworks_account_id() is None
    # Fallback verify path may trigger a second simple parse for api_key; ensure at least one call
    assert mock_parse_simple.call_count >= 1


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})
def test_get_account_id_from_ini_default_section_success(mock_parse_simple, mock_ConfigParser_class, mock_path_exists):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.return_value = [str(AUTH_INI_FILE)]

    def has_option_logic(section, option):
        if section == "fireworks":
            return False
        return section == mock_parser_instance.default_section and option == "account_id"

    def get_logic(section, option):
        if section == mock_parser_instance.default_section and option == "account_id":
            return INI_ACCOUNT_ID
        raise configparser.NoOptionError(option, section)

    mock_parser_instance.has_option.side_effect = has_option_logic
    mock_parser_instance.get.side_effect = get_logic
    mock_parser_instance.__contains__.side_effect = lambda item: item != "fireworks"
    with patch("builtins.open", mock_open(read_data="account_id = ini_account_id_def")):
        assert get_fireworks_account_id() == INI_ACCOUNT_ID
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
# We don't mock ConfigParser here because we are testing the _parse_simple_auth_file path directly
def test_get_account_id_from_ini_simple_parsing_success(
    mock_path_exists,
):  # Renamed from fallback_parsing
    file_content = f"account_id = {INI_ACCOUNT_ID}\nother_key = value"
    with patch("eval_protocol.auth.open", mock_open(read_data=file_content), create=True):
        # Ensure configparser path is not taken or also returns None for account_id
        with patch("configparser.ConfigParser") as mock_ConfigParser_class_inner:
            mock_parser_instance = mock_ConfigParser_class_inner.return_value
            mock_parser_instance.read.return_value = []
            mock_parser_instance.has_option.return_value = False
            assert get_fireworks_account_id() == INI_ACCOUNT_ID


# --- Tests for error handling ---


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Simple parse finds nothing
def test_get_api_key_ini_parse_error(mock_parse_simple, mock_ConfigParser_class, mock_path_exists, caplog):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = configparser.Error("Mocked Parsing Error")

    with patch("builtins.open", mock_open(read_data="malformed ini content")):
        assert get_fireworks_api_key() is None
    assert "Configparser error reading" in caplog.text
    assert "Mocked Parsing Error" in caplog.text
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Simple parse finds nothing
def test_get_account_id_ini_parse_error(mock_parse_simple, mock_ConfigParser_class, mock_path_exists, caplog):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = configparser.Error("Mocked Parsing Error")

    with patch("builtins.open", mock_open(read_data="malformed ini content")):
        assert get_fireworks_account_id() is None
    assert "Configparser error reading" in caplog.text
    assert "Mocked Parsing Error" in caplog.text
    # Fallback verify path may trigger a second simple parse for api_key; ensure at least one call
    assert mock_parse_simple.call_count >= 1


@patch("pathlib.Path.exists", return_value=True)
@patch("configparser.ConfigParser")
@patch("eval_protocol.auth._parse_simple_auth_file", return_value={})  # Simple parse finds nothing
def test_get_api_key_unexpected_error_reading_ini(
    mock_parse_simple, mock_ConfigParser_class, mock_path_exists, caplog
):
    mock_parser_instance = mock_ConfigParser_class.return_value
    mock_parser_instance.read.side_effect = Exception("Unexpected Read Error")

    with patch("builtins.open", mock_open(read_data="ini content")):
        assert get_fireworks_api_key() is None
    assert "Unexpected error reading" in caplog.text  # This comes from _get_credential_from_config_file
    assert "Unexpected Read Error" in caplog.text
    mock_parse_simple.assert_called_once_with(AUTH_INI_FILE)
