# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.usefixtures("compilation")
@pytest.mark.parametrize(
    "processor_cls",
    [sp.AddCategoryCounts, sp.ImputeMode],
)
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize(
    "vocabulary",
    [
        torch.tensor([10, 20]),
        StringTensor.from_list(["prefix", "a", "b", "suffix"])[1:3],
    ],
)
def test_categorical_transform_compilation(
    device: torch.device,
    processor_cls: type[sp.Processor],
    fullgraph: bool,
    vocabulary: torch.Tensor,
) -> None:
    vocabulary = vocabulary.to(device)
    context = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [1]], device=device),
            categories=(vocabulary,),
        ),
    )
    query = context.replace_blocks(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [-2]], device=device),
            categories=(vocabulary.clone(),),
        ),
    )
    processor = processor_cls().fit(context)
    compiled = torch.compile(processor.transform, fullgraph=fullgraph)
    expected = processor.transform(query)
    assert compiled(query).equal(expected)

    codes = query.categorical.code.clone()
    codes[0, 0] = 100
    with pytest.raises(ValueError, match="outside its category vocabulary"):
        compiled(
            query.replace_blocks(
                categorical=CategoricalTensor(
                    code=codes,
                    categories=query.categorical.categories,
                ),
            ),
        )

    reordered = vocabulary[torch.tensor([1, 0], device=device)]
    with pytest.raises(ValueError, match="match the fitted values and order"):
        compiled(
            query.replace_blocks(
                categorical=CategoricalTensor(
                    code=query.categorical.code,
                    categories=(reordered,),
                ),
            ),
        )
    assert compiled(query).equal(expected)
