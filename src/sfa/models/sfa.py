from typing import Any, Sequence

from torch import Tensor, nn

from .encoder import get_model

__all__ = ["SFA"]


class Classifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze(dim=1)
        return x


class Heatmaps(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        num_motion_channels: int,
        map_size: int,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.num_motion_channels = num_motion_channels
        self.map_size = map_size
        self.projection = nn.Linear(
            embed_dim,
            num_motion_channels * map_size * map_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = x.view(
            x.shape[0],
            self.num_landmarks,
            self.num_motion_channels,
            self.map_size,
            self.map_size,
        )
        return x


class SFA(nn.Module):
    def __init__(
        self,
        encoder_name: str = "marlin_base",
        heads: Sequence[str] = ("classifier", "heatmaps"),
        num_landmarks: int = 68,
        num_motion_channels: int = 1,
        map_size: int = 64,
        dropout: float = 0,
        **kwargs: Any,
    ):
        super().__init__()

        self.num_class_tokens = 0
        if "classifier" in heads:
            self.num_class_tokens += 1
        if "heatmaps" in heads:
            self.num_class_tokens += num_landmarks

        self.num_landmarks = num_landmarks
        self.encoder = get_model(
            encoder_name,
            num_class_tokens=self.num_class_tokens,
            **kwargs,
        )
        self.heads = nn.ModuleDict()
        for head_name in heads:
            if head_name == "classifier":
                self.heads[head_name] = Classifier(
                    self.encoder.embed_dim,
                    dropout=dropout,
                )
            elif head_name == "heatmaps":
                self.heads[head_name] = Heatmaps(
                    embed_dim=self.encoder.embed_dim,
                    num_landmarks=num_landmarks,
                    num_motion_channels=num_motion_channels,
                    map_size=map_size,
                )
            else:
                raise NotImplementedError(f"Invalid head name: {head_name}.")

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.encoder(x)
        outputs = {}
        if "heatmaps" in self.heads:
            lnd_tokens = x[:, -self.num_landmarks :]
            outputs["heatmaps"] = self.heads["heatmaps"](lnd_tokens)
        if "classifier" in self.heads:
            cls_token = x[:, [-self.num_class_tokens]]
            outputs["classifier"] = self.heads["classifier"](cls_token)
        return outputs
