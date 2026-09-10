"""From-scratch leukemia models for the ALL-IDB1 dataset."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    VGG11_BN_Weights,
    VGG16_BN_Weights,
    VGG19_BN_Weights,
    efficientnet_b0,
    mobilenet_v2,
    mobilenet_v3_large,
    resnet18,
    vgg11_bn,
    vgg16_bn,
    vgg19_bn,
)


@lru_cache(maxsize=16)
def _build_grid_adjacency(
    height: int,
    width: int,
    graph_neighbors: int,
) -> torch.Tensor:
    if graph_neighbors not in {4, 8}:
        raise ValueError("graph_neighbors must be 4 or 8.")

    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if graph_neighbors == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    node_count = height * width
    adjacency = torch.zeros(node_count, node_count)
    for row in range(height):
        for column in range(width):
            source = row * width + column
            adjacency[source, source] = 1.0
            for row_delta, column_delta in offsets:
                neighbor_row = row + row_delta
                neighbor_column = column + column_delta
                if 0 <= neighbor_row < height and 0 <= neighbor_column < width:
                    target = neighbor_row * width + neighbor_column
                    adjacency[source, target] = 1.0

    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


class OrthogonalSoftmaxLayer(nn.Module):
    """Masked linear classifier whose class weight vectors have disjoint support.

    The paper describes OSL as replacing the fully connected classification layer with
    a fixed binary mask M so selected classifier connections are kept throughout both
    training and testing. The paper's literal mask is diagonal: M[j, i] is one only
    when j equals i. Block and cyclic masks remain available only for ablations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mask_mode: str = "diagonal",
    ) -> None:
        super().__init__()
        if in_features < out_features:
            raise ValueError("OSL requires in_features >= out_features.")

        self.in_features = in_features
        self.out_features = out_features
        self.mask_mode = mask_mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("mask", self._build_mask(in_features, out_features, mask_mode))
        self.reset_parameters()

    @staticmethod
    def _build_mask(in_features: int, out_features: int, mask_mode: str) -> torch.Tensor:
        mask = torch.zeros(out_features, in_features)

        if mask_mode == "diagonal":
            for class_idx in range(out_features):
                mask[class_idx, class_idx] = 1.0
            return mask

        if mask_mode == "block":
            boundaries = torch.linspace(0, in_features, out_features + 1).round().long()
            for class_idx in range(out_features):
                start = int(boundaries[class_idx].item())
                end = int(boundaries[class_idx + 1].item())
                mask[class_idx, start:end] = 1.0
            return mask

        if mask_mode == "cyclic":
            for feature_idx in range(in_features):
                mask[feature_idx % out_features, feature_idx] = 1.0
            return mask

        raise ValueError("mask_mode must be one of: block, diagonal, cyclic")

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.linear(features, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_mode={self.mask_mode}"
        )


class ResNet18OSL(nn.Module):
    """Paper architecture: ResNet18 -> FC -> Dropout/ReLU x2 -> OSL."""

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        hidden_dim: int | None = None,
        dropout: float = 0.5,
        mask_mode: str = "diagonal",
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        base = resnet18(weights=weights)
        self.uses_pretrained_weights = weights is not None
        self.initialization_source = (
            f"ImageNet: {weights}" if weights is not None else "random: torchvision.resnet18(weights=None)"
        )

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

        classifier_features = num_classes if hidden_dim is None else hidden_dim
        if mask_mode == "diagonal" and classifier_features != num_classes:
            raise ValueError(
                "The paper's diagonal OSL requires the preceding FC layer to emit "
                "exactly one feature per class."
            )

        self.fc = nn.Linear(base.fc.in_features, classifier_features)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.osl = OrthogonalSoftmaxLayer(
            classifier_features,
            num_classes,
            bias=False,
            mask_mode=mask_mode,
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        hidden = self.fc(flattened)
        hidden = self.dropout1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.relu2(hidden)
        return hidden

    def forward_with_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_features(images)
        return self.osl(hidden), hidden

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits


class HybridCNNTransformer(nn.Module):
    """VGGNet-style CNN feature maps followed by a Transformer encoder."""

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        embed_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_dropout: float = 0.2,
        attention_dropout: float = 0.0,
        dropout: float = 0.5,
        cnn_backbone_name: str = "vgg11_bn",
    ) -> None:
        super().__init__()
        self.cnn_backbone_name = cnn_backbone_name
        self.cnn_backbone, cnn_feature_dim, weights = self._build_backbone(
            cnn_backbone_name, pretrained_backbone
        )
        self.uses_pretrained_weights = weights is not None
        self.initialization_source = (
            f"ImageNet: {weights}"
            if weights is not None
            else f"random: torchvision.{cnn_backbone_name}(weights=None)"
        )

        if freeze_backbone:
            for parameter in self.cnn_backbone.parameters():
                parameter.requires_grad = False

        self.projection = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            batch_first=True,
            dropout=transformer_dropout,
        )
        # MPS supports the residual and feed-forward dropout operations, but
        # not dropout inside scaled dot-product attention.
        encoder_layer.self_attn.dropout = attention_dropout
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _build_backbone(
        name: str, pretrained: bool
    ) -> tuple[nn.Module, int, object | None]:
        if name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
            return base.features, base.classifier[1].in_features, weights
        if name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_large(weights=weights)
            return base.features, base.classifier[0].in_features, weights
        if name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = mobilenet_v2(weights=weights)
            return base.features, base.classifier[1].in_features, weights
        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            return features, base.fc.in_features, weights
        if name in {"vggnet", "vgg11_bn"}:
            weights = VGG11_BN_Weights.DEFAULT if pretrained else None
            base = vgg11_bn(weights=weights)
            return base.features, 512, weights
        if name == "vgg16_bn":
            weights = VGG16_BN_Weights.DEFAULT if pretrained else None
            base = vgg16_bn(weights=weights)
            return base.features, 512, weights
        if name == "vgg19_bn":
            weights = VGG19_BN_Weights.DEFAULT if pretrained else None
            base = vgg19_bn(weights=weights)
            return base.features, 512, weights
        raise ValueError(
            "cnn_backbone_name must be one of: efficientnet_b0, "
            "mobilenet_v3_large, mobilenet_v2, resnet18, "
            "vggnet, vgg11_bn, vgg16_bn, vgg19_bn"
        )

    def _position_tokens(self, token_count: int) -> torch.Tensor:
        if token_count == self.position_embedding.shape[1]:
            return self.position_embedding

        cls_position = self.position_embedding[:, :1]
        spatial_position = self.position_embedding[:, 1:]
        old_size = int(math.sqrt(spatial_position.shape[1]))
        new_size = int(math.sqrt(token_count - 1))
        if new_size * new_size != token_count - 1:
            raise ValueError("Hybrid spatial token count must form a square grid.")
        spatial_position = spatial_position.transpose(1, 2).reshape(
            1, -1, old_size, old_size
        )
        spatial_position = F.interpolate(
            spatial_position,
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        spatial_position = spatial_position.flatten(2).transpose(1, 2)
        return torch.cat((cls_position, spatial_position), dim=1)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn_backbone(images)
        tokens = self.projection(features)
        batch_size = tokens.shape[0]
        tokens = tokens.flatten(2).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self._position_tokens(tokens.shape[1])
        encoded = self.transformer_encoder(tokens)
        return self.output_norm(encoded[:, 0])

    def forward_with_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        return self.fc(self.dropout(features)), features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits


class SpatialGraphConvolution(nn.Module):
    """Message-passing layer over the CNN feature-map grid."""

    def __init__(self, embed_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.self_projection = nn.Linear(embed_dim, embed_dim)
        self.neighbor_projection = nn.Linear(embed_dim, embed_dim)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_tokens = torch.einsum("ij,bjd->bid", adjacency, tokens)
        message = self.self_projection(tokens) + self.neighbor_projection(neighbor_tokens)
        tokens = self.message_norm(tokens + self.message_dropout(F.gelu(message)))
        return self.output_norm(tokens + self.feed_forward(tokens))


class HybridCNNGNN(nn.Module):
    """CNN feature extractor followed by graph message passing over spatial tokens."""

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        embed_dim: int = 256,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.2,
        graph_neighbors: int = 8,
        dropout: float = 0.5,
        cnn_backbone_name: str = "vgg11_bn",
    ) -> None:
        super().__init__()
        self.cnn_backbone_name = cnn_backbone_name
        self.graph_neighbors = graph_neighbors
        self.cnn_backbone, cnn_feature_dim, weights = HybridCNNTransformer._build_backbone(
            cnn_backbone_name, pretrained_backbone
        )
        self.uses_pretrained_weights = weights is not None
        self.initialization_source = (
            f"ImageNet: {weights}"
            if weights is not None
            else f"random: torchvision.{cnn_backbone_name}(weights=None)"
        )

        if freeze_backbone:
            for parameter in self.cnn_backbone.parameters():
                parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.graph_layers = nn.ModuleList(
            SpatialGraphConvolution(embed_dim, dropout=gnn_dropout)
            for _ in range(gnn_layers)
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.cnn_backbone(images)
        projected = self.projection(feature_map)
        _, _, height, width = projected.shape
        adjacency = _build_grid_adjacency(
            height,
            width,
            self.graph_neighbors,
        ).to(device=projected.device, dtype=projected.dtype)

        tokens = projected.flatten(2).permute(0, 2, 1)
        for graph_layer in self.graph_layers:
            tokens = graph_layer(tokens, adjacency)

        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        return self.readout(torch.cat((mean_pool, max_pool), dim=1))

    def forward_with_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        return self.fc(features), features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits


def create_model(
    num_classes: int,
    name: str = "resnet18_osl",
    pretrained_backbone: bool = False,
    freeze_backbone: bool = False,
    hidden_dim: int | None = None,
    dropout: float = 0.5,
    mask_mode: str = "diagonal",
    embed_dim: int = 256,
    transformer_heads: int = 8,
    transformer_layers: int = 4,
    transformer_dropout: float = 0.2,
    attention_dropout: float = 0.0,
    cnn_backbone_name: str = "vgg11_bn",
    gnn_layers: int = 3,
    gnn_dropout: float = 0.2,
    graph_neighbors: int = 8,
) -> nn.Module:
    if name == "resnet18_osl":
        return ResNet18OSL(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mask_mode=mask_mode,
        )

    if name == "hybrid_cnn_transformer":
        return HybridCNNTransformer(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
            attention_dropout=attention_dropout,
            dropout=dropout,
            cnn_backbone_name=cnn_backbone_name,
        )

    if name == "hybrid_cnn_gnn":
        return HybridCNNGNN(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            graph_neighbors=graph_neighbors,
            dropout=dropout,
            cnn_backbone_name=cnn_backbone_name,
        )

    raise ValueError(
        "name must be one of: resnet18_osl, hybrid_cnn_transformer, hybrid_cnn_gnn"
    )
