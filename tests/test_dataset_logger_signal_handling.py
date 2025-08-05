import os
import signal
import tempfile
import time
import json
from unittest.mock import patch
import pytest

from eval_protocol.dataset_logger.local_fs_dataset_logger_adapter import LocalFSDatasetLoggerAdapter
from eval_protocol.models import EvaluationRow


class TestLocalFSDatasetLoggerAdapterSignalHandling:
    """Test that the dataset logger properly handles termination signals."""

    def setup_method(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create a mock .eval_protocol directory
        os.makedirs(".eval_protocol/datasets", exist_ok=True)

    def teardown_method(self):
        """Clean up after tests."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_signal_handlers_registered(self):
        """Test that signal handlers are properly registered."""
        logger = LocalFSDatasetLoggerAdapter()
        
        # Check that signal handlers are registered
        assert signal.getsignal(signal.SIGINT) != signal.SIG_DFL
        assert signal.getsignal(signal.SIGTERM) != signal.SIG_DFL

    def test_index_saved_on_sigint(self):
        """Test that index is saved when SIGINT is received."""
        logger = LocalFSDatasetLoggerAdapter()
        
        # Create a test row and log it
        test_row = EvaluationRow(
            id="test_id_1",
            prompt="test prompt",
            response="test response",
            score=0.8
        )
        logger.log(test_row)
        
        # Verify index is dirty
        assert logger._index_dirty is True
        
        # Simulate SIGINT
        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                # Get the signal handler
                sigint_handler = signal.getsignal(signal.SIGINT)
                # Call it directly
                sigint_handler(signal.SIGINT, None)
                
                # Verify flush was called and exit was called
                assert not logger._index_dirty  # Index should be flushed
                mock_exit.assert_called_once_with(0)
                mock_print.assert_called()

    def test_index_saved_on_sigterm(self):
        """Test that index is saved when SIGTERM is received."""
        logger = LocalFSDatasetLoggerAdapter()
        
        # Create a test row and log it
        test_row = EvaluationRow(
            id="test_id_2",
            prompt="test prompt",
            response="test response",
            score=0.9
        )
        logger.log(test_row)
        
        # Verify index is dirty
        assert logger._index_dirty is True
        
        # Simulate SIGTERM
        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                # Get the signal handler
                sigterm_handler = signal.getsignal(signal.SIGTERM)
                # Call it directly
                sigterm_handler(signal.SIGTERM, None)
                
                # Verify flush was called and exit was called
                assert not logger._index_dirty  # Index should be flushed
                mock_exit.assert_called_once_with(0)
                mock_print.assert_called()

    def test_force_flush_method(self):
        """Test the force_flush method."""
        logger = LocalFSDatasetLoggerAdapter()
        
        # Create a test row and log it
        test_row = EvaluationRow(
            id="test_id_3",
            prompt="test prompt",
            response="test response",
            score=0.7
        )
        logger.log(test_row)
        
        # Verify index is dirty
        assert logger._index_dirty is True
        
        # Force flush
        logger.force_flush()
        
        # Verify index is no longer dirty
        assert logger._index_dirty is False
        
        # Verify index file exists and contains the expected data
        assert os.path.exists(logger.index_path)
        with open(logger.index_path, 'r') as f:
            index_data = json.load(f)
        assert "test_id_3" in index_data

    def test_flush_error_handling(self):
        """Test that flush errors are handled gracefully."""
        logger = LocalFSDatasetLoggerAdapter()
        
        # Create a test row and log it
        test_row = EvaluationRow(
            id="test_id_4",
            prompt="test prompt",
            response="test response",
            score=0.6
        )
        logger.log(test_row)
        
        # Mock _save_index_to_disk to raise an exception
        with patch.object(logger, '_save_index_to_disk', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                logger.flush()
                
                # Verify error was logged but not raised
                mock_print.assert_called()
                assert "Warning: Failed to flush index to disk" in mock_print.call_args[0][0]
                
                # Verify index is still dirty (so it can be retried)
                assert logger._index_dirty is True 