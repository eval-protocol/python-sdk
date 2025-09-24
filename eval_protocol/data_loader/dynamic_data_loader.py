from collections.abc import Callable, Sequence
from dataclasses import dataclass

from eval_protocol.data_loader.models import (
    DataLoaderResult,
    DataLoaderVariant,
    EvaluationDataLoader,
)
from eval_protocol.models import EvaluationRow


@dataclass(kw_only=True)
class DynamicDataLoader(EvaluationDataLoader):
    """Data loader for dynamic data generation."""

    generators: Sequence[Callable[[], list[EvaluationRow]]]
    """Dynamic data generation functions. These callables are invoked each time data
    needs to be loaded, allowing for dynamic data generation, lazy loading, or data that
    changes between evaluation runs. Each function should return a list of EvaluationRow
    objects. This is useful for scenarios like generating test data on-the-fly, loading
    data from external sources, or creating data with randomized elements for robust testing."""

    def variants(self) -> Sequence[DataLoaderVariant]:
        variants: Sequence[DataLoaderVariant] = []
        for generator in self.generators:

            def _load() -> DataLoaderResult:
                resolved_rows = generator()
                return DataLoaderResult(
                    rows=resolved_rows,
                    type=self.__class__.__name__,
                    variant_id=generator.__name__,
                    variant_description=generator.__doc__,
                )

            variants.append(_load)

        return variants
