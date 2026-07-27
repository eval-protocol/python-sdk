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

    # Build a map of saved rows by row_id for order-independent comparison
    # (read() now returns rows in descending order by insertion time)
    saved_by_row_id = {r.input_metadata.row_id: r for r in saved_rows}

    # Verify each row matches the original (order-independent)
    for i, original_row in enumerate(rows):
        row_id = f"row_{i}"
        saved_row = saved_by_row_id[row_id]
        assert original_row.messages == saved_row.messages
        assert original_row.input_metadata == saved_row.input_metadata
        assert saved_row.input_metadata.row_id == row_id


def test_read_with_invocation_ids_filter():
    """Test filtering rows by invocation_ids."""
    from eval_protocol.models import ExecutionMetadata

    db_path = get_db_path("test_read_with_invocation_ids_filter")
    # delete the db file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SqliteEvaluationRowStore(db_path=db_path)
    logger = SqliteDatasetLoggerAdapter(store=store)

    # Create rows with different invocation_ids
    inv_ids = ["inv-alpha", "inv-beta", "inv-gamma"]
    for i, inv_id in enumerate(inv_ids):
        messages = [Message(role="user", content=f"Hello {inv_id}")]
        input_metadata = InputMetadata(row_id=f"row_{i}")
        execution_metadata = ExecutionMetadata(invocation_id=inv_id)
        row = EvaluationRow(
            input_metadata=input_metadata,
            messages=messages,
            execution_metadata=execution_metadata,
        )
        logger.log(row)

    # Test 1: Read all (no filter)
    all_rows = logger.read()
    assert len(all_rows) == 3

    # Test 2: Filter by single invocation_id
    filtered_rows = logger.read(invocation_ids=["inv-alpha"])
    assert len(filtered_rows) == 1
    assert filtered_rows[0].execution_metadata.invocation_id == "inv-alpha"

    # Test 3: Filter by multiple invocation_ids
    filtered_rows = logger.read(invocation_ids=["inv-alpha", "inv-gamma"])
    assert len(filtered_rows) == 2
    inv_ids_found = {r.execution_metadata.invocation_id for r in filtered_rows}
    assert inv_ids_found == {"inv-alpha", "inv-gamma"}

    # Test 4: Filter by non-existent invocation_id
    filtered_rows = logger.read(invocation_ids=["inv-nonexistent"])
    assert len(filtered_rows) == 0


def test_read_with_limit():
    """Test limiting the number of rows returned."""
    db_path = get_db_path("test_read_with_limit")
    # delete the db file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SqliteEvaluationRowStore(db_path=db_path)
    logger = SqliteDatasetLoggerAdapter(store=store)

    # Create 10 rows
    for i in range(10):
        messages = [Message(role="user", content=f"Hello {i}")]
        input_metadata = InputMetadata(row_id=f"row_{i}")
        row = EvaluationRow(input_metadata=input_metadata, messages=messages)
        logger.log(row)

    # Test with limit
    limited_rows = logger.read(limit=3)
    assert len(limited_rows) == 3

    # Verify we got the most recent rows (inserted last, returned first)
    row_ids = [r.input_metadata.row_id for r in limited_rows]
    assert row_ids == ["row_9", "row_8", "row_7"]  # Most recent first
