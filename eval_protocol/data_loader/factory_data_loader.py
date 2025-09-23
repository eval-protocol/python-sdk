from collections.abc import Callable, Sequence

from eval_protocol.data_loader.models import (
    DataLoaderContext,
    DataLoaderResult,
    DataLoaderVariant,
    EvaluationDataLoader,
)
from eval_protocol.models import EvaluationRow


class FactoryDataLoader(EvaluationDataLoader):
    """Data loader for factory of list[EvaluationRow]"""

    description: str | None = None
    """Optional human-readable description of this data loader. Provides additional
    context about the data source, purpose, or any special characteristics. Used for
    documentation and debugging purposes. If not provided, the variant_id will be used instead."""

    factory: Sequence[Callable[[], list[EvaluationRow]]]
    """Factory function that generates evaluation rows dynamically. This callable
    is invoked each time data needs to be loaded, allowing for dynamic data generation,
    lazy loading, or data that changes between evaluation runs. The factory should return
    a list of EvaluationRow objects. This is useful for scenarios like generating test
    data on-the-fly, loading data from external sources, or creating data with randomized
    elements for robust testing."""

    def variants(self) -> Sequence[DataLoaderVariant]:
        variants: Sequence[DataLoaderVariant] = []
        for factory in self.factory:

            def _load(ctx: DataLoaderContext) -> DataLoaderResult:
                resolved_rows = factory()
                return DataLoaderResult(
                    rows=resolved_rows,
                    num_rows=len(resolved_rows),
                    type="factory",
                    variant_id=ctx.variant_id,
                    variant_description=ctx.variant_description,
                )

            variants.append(
                DataLoaderVariant(
                    id=factory.__name__,
                    description=factory.__doc__,
                    loader=_load,
                )
            )

        return variants
