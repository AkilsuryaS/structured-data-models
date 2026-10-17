# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sdm.models.timestral.config import LAYER_KIND_AXES
from sdm.nn import SDPA, RMSNorm, SwiGLU


class _SinCosRoPE(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, base: float, rotary_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.rotary_dim = rotary_dim
        self.register_buffer(
            "inv_freq", self._build_inv_freq(None), persistent=False
        )

    def _build_inv_freq(self, device: torch.device | None) -> Tensor:
        # Partial rotation pairs channels across the full head halves.
        inv_freq = torch.zeros(
            self.dim // 2, dtype=torch.float32, device=device
        )
        idx = torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float32, device=device
        )
        inv_freq[: self.rotary_dim // 2] = 1.0 / self.base ** (
            idx / self.rotary_dim
        )
        return inv_freq

    def rebuild_inv_freq(self) -> None:
        self.inv_freq = self._build_inv_freq(self.inv_freq.device)

    @torch.no_grad()
    def forward(
        self, x: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


@dataclass(frozen=True)
class _TwoStreamMask:
    mask: Tensor
    enc_len: int


class _Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_time: bool,
        use_bias: bool,
        gate: str,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.sdpa = SDPA(num_query_heads=num_heads)
        self.qkv_lin = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_lin = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.gate_lin = (
            nn.Linear(embed_dim, embed_dim, bias=use_bias)
            if gate == "sdpa_sigmoid"
            else None
        )

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        qkv = self.qkv_lin(x).unflatten(-1, (3, self.num_heads, self.head_dim))
        return qkv.unbind(-3)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        return self.sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=is_causal,
        ).flatten(-2)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | _TwoStreamMask | None,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        rows, num_patches, _ = x.shape
        inputs = x if self.is_time else x.permute(1, 0, 2)
        q, k, v = self._qkv(inputs)
        if self.is_time:
            q, k = _apply_rotary(q, k, cos, sin)
        if isinstance(mask, _TwoStreamMask):
            n = mask.enc_len
            out_enc = self._attend(
                query=q[:, :n],
                key=k[:, :n],
                value=v[:, :n],
                mask=mask.mask[:, :, :n, :n],
            )
            out_dec = self._attend(q[:, n:], k, v, mask.mask[:, :, n:, :])
            out = torch.cat([out_enc, out_dec], dim=1)
        else:
            out = self._attend(
                query=q,
                key=k,
                value=v,
                mask=mask,
                is_causal=self.is_time and mask is None,
            )
        if self.gate_lin is not None:
            out = out * self.gate_lin(inputs).sigmoid()
        if self.is_time:
            out = out.reshape(rows, num_patches, self.embed_dim)
        else:
            out = out.reshape(num_patches, rows, self.embed_dim).permute(
                1, 0, 2
            )
        return self.out_lin(out)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=norm_eps) for _ in axes]
        )
        self.attns = nn.ModuleList(
            [
                _Attention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    is_time=is_time,
                    use_bias=attn_bias,
                    gate=time_attn_gate if is_time else variate_attn_gate,
                )
                for is_time in axes
            ]
        )
        self.mlp_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim, bias=mlp_bias)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | _TwoStreamMask | None,
        variate_mask: Tensor | None,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(
                norm(x), time_mask if attn.is_time else variate_mask, cos, sin
            )
        return x + self.mlp(self.mlp_norm(x))


class _Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    kind=kind,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_eps=norm_eps,
                    attn_bias=attn_bias,
                    mlp_bias=mlp_bias,
                    time_attn_gate=time_attn_gate,
                    variate_attn_gate=variate_attn_gate,
                )
                for kind in block_schedule
            ]
        )
        self.rope = _SinCosRoPE(
            dim=embed_dim // num_heads, base=rope_theta, rotary_dim=rope_dim
        )
        self.out_norm = RMSNorm(embed_dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | _TwoStreamMask | None,
        variate_mask: Tensor | None,
        positions: Tensor,
    ) -> Tensor:
        cos, sin = self.rope(x, positions)
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.out_norm(x)
