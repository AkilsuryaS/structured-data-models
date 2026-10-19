# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.timestral.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("column_every", [None, 1, 2, 4])
def test_icl_block(device: torch.device, column_every: int | None) -> None:
    block = ICLBlock(
        channels=16,
        num_layers=4,
        num_heads=2,
        hidden_channels=24,
        column_every=column_every,
        device=device,
    )
    x = torch.randn(2, 3, 5, 16, device=device)
    row_mask = torch.ones(5, 5, device=device, dtype=torch.bool).tril()[None]
    out = block(x, row_mask=row_mask, col_mask=None)
    assert out.shape == x.shape
    assert out.isfinite().all()
    out.square().mean().backward()
    assert all(
        p.grad is not None for p in block.parameters() if p.requires_grad
    )

    with torch.inference_mode():
        torch.testing.assert_close(
            block(x, row_mask=row_mask, col_mask=None), out
        )


@withCUDA
def test_icl_block_masks_and_positions(device: torch.device) -> None:
    block = ICLBlock(
        channels=16,
        num_layers=4,
        num_heads=2,
        hidden_channels=24,
        column_every=2,
        device=device,
    )
    # Exercise masking with nonzero attention output projections.
    for module in block.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    x = torch.randn(2, 3, 5, 16, device=device)
    pos = torch.arange(5, device=device).expand(2, 3, 5)
    row_mask = torch.ones(5, 5, device=device, dtype=torch.bool).tril()[None]
    col_mask = torch.eye(3, device=device, dtype=torch.bool)[None]
    with torch.inference_mode():
        out = block(x, row_mask=row_mask, col_mask=col_mask)
        torch.testing.assert_close(
            block(x, row_mask=row_mask, col_mask=col_mask, pos=pos), out
        )
        # Masked columns and future rows must not influence this prediction.
        changed = x.clone()
        changed[:, 1:] += torch.randn_like(changed[:, 1:])
        changed[:, 0, 3:] += torch.randn_like(changed[:, 0, 3:])
        torch.testing.assert_close(
            block(changed, row_mask=None, col_mask=col_mask, pos=pos)[
                :, 0, :3
            ],
            out[:, 0, :3],
        )
        # The supplied mask can allow bidirectional row attention.
        full_mask = torch.ones_like(row_mask)
        full = block(x, row_mask=full_mask, col_mask=col_mask, pos=pos)
        torch.testing.assert_close(
            block(x, row_mask=None, col_mask=col_mask, pos=pos), out
        )
        assert not torch.allclose(full[:, 0, :3], out[:, 0, :3])
        assert not torch.allclose(
            block(x, row_mask=row_mask, col_mask=None)[:, 0], out[:, 0]
        )
        assert not torch.allclose(
            block(x, row_mask=row_mask, col_mask=col_mask, pos=pos.square()),
            out,
        )
        torch.testing.assert_close(
            block(x[:1], row_mask=row_mask, col_mask=col_mask), out[:1]
        )


@withCUDA
@pytest.mark.parametrize("strided", [False, True])
def test_icl_block_inference(device: torch.device, strided: bool) -> None:
    block = ICLBlock(
        channels=16,
        num_layers=4,
        num_heads=2,
        hidden_channels=24,
        column_every=2,
        device=device,
    )
    for module in block.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    x = torch.randn(2, 3, 5, 16, device=device)
    if strided:
        x = x.transpose(-3, -2)
    original = x.clone()
    pos = torch.arange(x.size(-2), device=device).square()
    expected = block(x, row_mask=None, col_mask=None, pos=pos)
    with torch.inference_mode():
        out = block(x, row_mask=None, col_mask=None, pos=pos)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
