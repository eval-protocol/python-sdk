import os

from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore
from eval_protocol.models import EvaluationRow, InputMetadata, Message


def get_db_path(test_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{test_name}.db")


def test_update_log_and_read():
    db_path = get_db_path("test_update_log_and_read")
    # delete the db file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SqliteEvaluationRowStore(db_path=db_path)
    messages = [Message(role="user", content="Hello")]
    input_metadata = InputMetadata(row_id="1")
    row = EvaluationRow(input_metadata=input_metadata, messages=messages)
    store.upsert_row(data=row.model_dump(exclude_none=True, mode="json"))

    row.messages.append(Message(role="assistant", content="Hello"))

    logger = SqliteDatasetLoggerAdapter(store=store)
    logger.log(row)
    saved = logger.read(row.execution_metadata.rollout_id)[0]
    assert row.messages == saved.messages
    assert row.input_metadata == saved.input_metadata


def test_create_log_and_read():
    db_path = get_db_path("test_create_log_and_read")
    # delete the db file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SqliteEvaluationRowStore(db_path=db_path)

    logger = SqliteDatasetLoggerAdapter(store=store)
    messages = [Message(role="user", content="Hello")]
    input_metadata = InputMetadata(row_id="1")
    row = EvaluationRow(input_metadata=input_metadata, messages=messages)

    logger.log(row)
    saved = logger.read(rollout_id=row.execution_metadata.rollout_id)[0]
    assert row.messages == saved.messages
    assert row.input_metadata == saved.input_metadata


def test_create_multiple_logs_and_read_all():
    db_path = get_db_path("test_create_multiple_logs_and_read_all")
    # delete the db file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SqliteEvaluationRowStore(db_path=db_path)
    logger = SqliteDatasetLoggerAdapter(store=store)

    # Create multiple evaluation rows with different row_ids
    rows = []
    for i in range(3):
        messages = [Message(role="user", content=f"Hello {i}")]
        input_metadata = InputMetadata(row_id=f"row_{i}")
        row = EvaluationRow(input_metadata=input_metadata, messages=messages)
        rows.append(row)

        # Log each row
        logger.log(row)

    # Read all logs (without specifying row_id)
    saved_rows = logger.read()

    # Verify we got all 3 rows back
    assert len(saved_rows) == 3

    # Verify each row matches the original
    for i, original_row in enumerate(rows):
        saved_row = saved_rows[i]
        assert original_row.messages == saved_row.messages
        assert original_row.input_metadata == saved_row.input_metadata
        assert original_row.input_metadata.row_id == f"row_{i}"
