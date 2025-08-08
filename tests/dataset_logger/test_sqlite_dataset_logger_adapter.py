from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter
from eval_protocol.models import EvaluationRow, InputMetadata, Message


def test_log_and_read():
    logger = SqliteDatasetLoggerAdapter()
    messages = [Message(role="user", content="Hello")]
    input_metadata = InputMetadata(row_id="1")
    row = EvaluationRow(input_metadata=input_metadata, messages=messages)
    logger.log(row)
    saved = logger.read(row_id="1")[0]
    assert row.messages == saved.messages
    assert row.input_metadata == saved.input_metadata
