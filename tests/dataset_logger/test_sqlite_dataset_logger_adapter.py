import os

from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore
from eval_protocol.models import EvaluationRow, InputMetadata, Message

DB_PATH = os.path.join(os.path.dirname(__file__), "test.db")


def test_update_log_and_read():
    store = SqliteEvaluationRowStore(db_path=DB_PATH)
    messages = [Message(role="user", content="Hello")]
    input_metadata = InputMetadata(row_id="1")
    row = EvaluationRow(input_metadata=input_metadata, messages=messages)
    store.upsert_row(row_id="1", data=row.model_dump(exclude_none=True, mode="json"))

    row.messages.append(Message(role="assistant", content="Hello"))

    logger = SqliteDatasetLoggerAdapter()
    logger.log(row)
    saved = logger.read(row_id="1")[0]
    assert row.messages == saved.messages
    assert row.input_metadata == saved.input_metadata


def test_create_log_and_read():
    store = SqliteEvaluationRowStore(db_path=DB_PATH)
    store.delete_row(row_id="1")

    logger = SqliteDatasetLoggerAdapter()
    messages = [Message(role="user", content="Hello")]
    input_metadata = InputMetadata(row_id="1")
    row = EvaluationRow(input_metadata=input_metadata, messages=messages)

    logger.log(row)
    saved = logger.read(row_id="1")[0]
    assert row.messages == saved.messages
    assert row.input_metadata == saved.input_metadata
