# Adapted from https://github.com/ControlNet/MARLIN

from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from .modules import PatchEmbedding, PositionalEmbedding, TransformerBlock

__all__ = ["get_model"]


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_class_tokens: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qv_bias: bool = True,
        scale: float | None = None,
        attention_head_dim: int | None = None,
        norm_layer: Callable[..., nn.Module] | None = partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        activation: Callable[..., nn.Module] | None = nn.GELU,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(
            patch_size=(tubelet_size, patch_size, patch_size),
            hidden_dim=embed_dim,
        )

        self.num_class_tokens = num_class_tokens
        if self.num_class_tokens > 0:
            self.class_tokens = nn.Parameter(
                torch.zeros(1, num_class_tokens, embed_dim)
            )

        num_tokens = (
            num_class_tokens
            + (num_frames // tubelet_size) * (image_size // patch_size) ** 2
        )
        self.pos_embedding = PositionalEmbedding(num_tokens, embed_dim)

        blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                qv_bias=qv_bias,
                scale=scale,
                attention_head_dim=attention_head_dim,
                norm_layer=norm_layer,
                activation=activation,
            )
            for _ in range(num_layers)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)
        if self.num_class_tokens > 0:
            class_tokens = self.class_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x, class_tokens], dim=1)
        x = self.pos_embedding(x)
        x = self.blocks(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


def get_model(model_name: str, **kwargs) -> VisionTransformer:
    if model_name == "marlin_small":
        model = VisionTransformer(
            patch_size=16,
            tubelet_size=2,
            num_layers=12,
            embed_dim=384,
            num_heads=6,
            **kwargs,
        )
    elif model_name == "marlin_base":
        model = VisionTransformer(
            patch_size=16,
            tubelet_size=2,
            num_layers=12,
            embed_dim=768,
            num_heads=12,
            **kwargs,
        )
    elif model_name == "marlin_large":
        model = VisionTransformer(
            patch_size=16,
            tubelet_size=2,
            num_layers=24,
            embed_dim=1024,
            num_heads=16,
            **kwargs,
        )

    return model
