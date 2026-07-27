from eval_protocol.data_loader import DynamicDataLoader
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor


class InMemoryLogger(DatasetLogger):
    def log(self, row: EvaluationRow) -> None:
        return None

    def read(self) -> list[EvaluationRow]:
        return []


class StopAfterPreprocess(Exception):
    pass


class StopAfterPreprocessRolloutProcessor(NoOpRolloutProcessor):
    def setup(self) -> None:
        raise StopAfterPreprocess("Stop after preprocessing for focused test assertions")


def _build_rows() -> list[EvaluationRow]:
    return [
        EvaluationRow(
            messages=[
                Message(role="user", content="question"),
                Message(role="assistant", content="answer"),
            ]
        )
    ]


async def test_preprocess_fn_runs_with_data_loader_without_loader_preprocess():
    call_count = {"decorator_preprocess": 0}

    def decorator_preprocess(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        call_count["decorator_preprocess"] += 1
        return rows

    data_loader = DynamicDataLoader(generators=[_build_rows])

    @evaluation_test(
        data_loaders=data_loader,
        preprocess_fn=decorator_preprocess,
        rollout_processor=StopAfterPreprocessRolloutProcessor(),
        logger=InMemoryLogger(),
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        row.evaluation_result = EvaluateResult(score=1.0, reason="ok")
        return row

    try:
        await eval_fn(data_loaders=data_loader)
    except StopAfterPreprocess:
        pass

    assert call_count["decorator_preprocess"] == 1


async def test_preprocess_fn_not_double_applied_when_data_loader_preprocess_exists():
    call_count = {"loader_preprocess": 0, "decorator_preprocess": 0}

    def loader_preprocess(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        call_count["loader_preprocess"] += 1
        return rows

    def decorator_preprocess(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        call_count["decorator_preprocess"] += 1
        return rows

    data_loader = DynamicDataLoader(generators=[_build_rows], preprocess_fn=loader_preprocess)

    @evaluation_test(
        data_loaders=data_loader,
        preprocess_fn=decorator_preprocess,
        rollout_processor=StopAfterPreprocessRolloutProcessor(),
        logger=InMemoryLogger(),
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        row.evaluation_result = EvaluateResult(score=1.0, reason="ok")
        return row

    try:
        await eval_fn(data_loaders=data_loader)
    except StopAfterPreprocess:
        pass

    assert call_count["loader_preprocess"] == 1
    assert call_count["decorator_preprocess"] == 0
