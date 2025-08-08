# Adapted from https://github.com/ControlNet/MARLIN

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "Attention",
    "LinearNormActivation",
    "MLP",
    "PatchEmbedding",
    "PositionalEmbedding",
    "TransformerBlock",
]


class LinearNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation: Callable[..., nn.Module] | None = None,
        normalization_after_activation: bool = False,
    ):
        layers = OrderedDict()
        layers["linear"] = nn.Linear(in_channels, out_channels, bias=bias)

        norm_act = []
        if norm_layer is not None:
            norm_act.append({"normalization": norm_layer(out_channels)})
        if activation is not None:
            norm_act.append({"activation": activation()})
        if normalization_after_activation:
            norm_act = norm_act[::-1]

        for layer in norm_act:
            for k, v in layer.items():
                layers[k] = v

        if dropout > 0:
            layers["dropout"] = nn.Dropout(dropout)

        super().__init__(layers)


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        bias: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation: Callable[..., nn.Module] | None = None,
        normalization_after_activation: bool = False,
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(
                LinearNormActivation(
                    in_dim,
                    hidden_dim,
                    bias=bias,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    activation=activation,
                    normalization_after_activation=normalization_after_activation,
                )
            )
            in_dim = hidden_dim
        layers.append(
            LinearNormActivation(
                in_dim, hidden_channels[-1], bias=bias, dropout=dropout
            )
        )

        super().__init__(*layers)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qv_bias: bool = False,
        scale: float | None = None,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        attention_head_dim: int | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        if attention_head_dim is not None:
            head_dim = attention_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, all_head_dim * 3, bias=False)
        if qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(all_head_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        qv_bias: bool,
        scale: float | None,
        attention_head_dim: int | None,
        norm_layer: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
    ):
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads,
            qv_bias=qv_bias,
            scale=scale,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            attention_head_dim=attention_head_dim,
        )

        self.norm2 = norm_layer(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            embed_dim,
            [mlp_dim, embed_dim],
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int],
        hidden_dim: int,
        strides: int | tuple[int, int, int] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        assert len(patch_size) == 3, len(patch_size)

        if strides is None:
            strides = patch_size
        elif isinstance(strides, int):
            strides = (strides, strides, strides)

        self.projection = nn.Conv3d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=strides,
        )
        if norm_layer is not None:
            self.normalization = norm_layer(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1])
        if self.normalization is not None:
            x = self.normalization(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        super().__init__()

        embedding = self.get_sinusoid_encoding_table(num_tokens, embed_dim)
        self.embedding = nn.Parameter(
            embedding.unsqueeze(0),
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.embedding

    @staticmethod
    def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2
                * torch.div(torch.arange(d_hid), 2, rounding_mode="trunc")
                / d_hid
            )

        sinusoid_table = torch.stack(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.float()
