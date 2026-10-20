"""Experimental ordinary tensor container, not an SDM TableTensor replacement.

Only numerical/categorical blocks, row/column cat and batch stack are modeled.
The container is not a Tensor: table autograd, buffers, dispatcher composition,
collation and existing serialization require API migration. Compiled cat/stack
use the explicit classmethods: torch.cat/stack dispatch works only eagerly.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils import _pytree

from sdm import Stype


@dataclass(frozen=True)
class CategoricalData:
    code: Tensor
    _categories: tuple[Tensor, ...]

    @property
    def categories(self) -> tuple[Tensor, ...]:
        return tuple(self._categories)


@dataclass(frozen=True)
class TableData:
    numerical: Tensor
    categorical: CategoricalData
    _names: tuple[tuple[str, ...], tuple[str, ...]]

    @property
    def columns(self) -> dict[Stype, tuple[str, ...]]:
        return {
            Stype.numerical: self._names[0],
            Stype.categorical: self._names[1],
        }

    def to(self, *args: Any, **kwargs: Any) -> TableData:
        numerical = self.numerical.to(*args, **kwargs)
        categorical = CategoricalData(
            self.categorical.code.to(device=numerical.device),
            tuple(
                c.to(device=numerical.device)
                for c in self.categorical.categories
            ),
        )
        return TableData(numerical, categorical, self._names)

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func not in (torch.cat, torch.stack):
            return NotImplemented
        kwargs = {} if kwargs is None else kwargs
        tables = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        if set(kwargs) - {"dim"}:
            return NotImplemented
        return cls._combine(func, tables, dim)

    @classmethod
    def cat(cls, tables: Sequence[TableData], dim: int = 0) -> TableData:
        return cls._combine(torch.cat, tables, dim)

    @classmethod
    def stack(cls, tables: Sequence[TableData], dim: int = 0) -> TableData:
        return cls._combine(torch.stack, tables, dim)

    @classmethod
    def _combine(
        cls,
        func: Callable[..., Tensor],
        tables: Sequence[TableData],
        dim: int,
    ) -> TableData:
        first = tables[0]
        if func is torch.cat and dim == -1:
            categories = tuple(
                c for t in tables for c in t.categorical.categories
            )
            names = (
                tuple(n for t in tables for n in t._names[0]),
                tuple(n for t in tables for n in t._names[1]),
            )
        else:
            if dim != 0:
                raise NotImplementedError(
                    "Only row cat and batch stack are modeled"
                )
            categories = first.categorical.categories
            names = first._names
            for table in tables[1:]:
                if table._names != names or len(
                    table.categorical.categories
                ) != len(categories):
                    raise ValueError(
                        "Row operations require matching column metadata"
                    )
                if any(
                    a is not b
                    for a, b in zip(categories, table.categorical.categories)
                ):
                    raise ValueError(
                        "Row operations require shared vocabulary tensors"
                    )
        return cls(
            func(tuple(t.numerical for t in tables), dim=dim),
            CategoricalData(
                func(tuple(t.categorical.code for t in tables), dim=dim),
                categories,
            ),
            names,
        )


def _unflatten_categories(
    values: Iterable[Any], context: Any
) -> CategoricalData:
    code, *categories = values
    return CategoricalData(code, tuple(categories))


def _unflatten_table(values: Iterable[Any], context: Any) -> TableData:
    numerical, categorical = values
    return TableData(numerical, categorical, context)


_pytree.register_pytree_node(
    CategoricalData,
    lambda value: ([value.code, *value.categories], None),
    _unflatten_categories,
)
_pytree.register_pytree_node(
    TableData,
    lambda value: ([value.numerical, value.categorical], value._names),
    _unflatten_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("eager", "inductor"), default="eager"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    table = TableData(
        torch.arange(6.0).reshape(3, 2).requires_grad_(),
        CategoricalData(
            torch.tensor([[0], [1], [-1]]), (torch.tensor([10, 20]),)
        ),
        (("a", "b"), ("kind",)),
    )

    def transform(value: TableData) -> tuple[TableData, Tensor]:
        value = TableData(value.numerical + 1, value.categorical, value._names)
        out = TableData.stack((TableData.cat((value, value)),) * 2)
        return out.to(dtype=torch.float64), out.numerical.sum()

    result, loss = torch.compile(
        transform, backend=args.backend, fullgraph=True
    )(table)
    loss.backward()
    print(result.columns, table.numerical.grad)


if __name__ == "__main__":
    main()
