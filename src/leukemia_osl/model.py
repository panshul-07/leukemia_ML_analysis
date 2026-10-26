
"""Dataset-specific pretrained CNN backbones with a dynamic graph network."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    inception_v3,
    Inception_V3_Weights,
)


# ==========================================
# 1. SPATIAL GRAPH CONSTRUCTION
# ==========================================

@lru_cache(maxsize=16)
def grid_adjacency(height, width, neighbors=8):

    if neighbors not in (4, 8):
        raise ValueError("graph_neighbors must be 4 or 8")

    offsets = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1)
    ]

    if neighbors == 8:
        offsets += [
            (-1, -1), (-1, 1),
            (1, -1), (1, 1)
        ]

    adj = torch.eye(
        height * width,
        dtype=torch.bool
    )

    for r in range(height):
        for c in range(width):
            for dr, dc in offsets:

                rr = r + dr
                cc = c + dc

                if 0 <= rr < height and 0 <= cc < width:
                    adj[
                        r * width + c,
                        rr * width + cc
                    ] = True

    return adj


# ==========================================
# 2. CNN FEATURE EXTRACTION
# ==========================================

class EfficientNetB3Features(nn.Module):

    def __init__(self, base):
        super().__init__()

        self.features = base.features

    def forward(self, images):

        low = None
        x = images

        for i, stage in enumerate(self.features):
            x = stage(x)

            if i == 5:
                low = x

        return low, x

    @property
    def gradcam_target(self):
        return self.features[-1]


class EfficientNetV2SFeatures(nn.Module):
    """Return 24x24 and 12x12 feature maps at the paper's 384px input."""

    def __init__(self, base):
        super().__init__()
        self.features = base.features

    def forward(self, images):
        low = None
        x = images
        for index, stage in enumerate(self.features):
            x = stage(x)
            if index == 5:
                low = x
        return low, x

    @property
    def gradcam_target(self):
        return self.features[-1]


class InceptionV3Features(nn.Module):
    """Return the final two spatial scales from an InceptionV3 backbone."""

    def __init__(self, base):
        super().__init__()
        self.Conv2d_1a_3x3 = base.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = base.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = base.Conv2d_2b_3x3
        self.maxpool1 = base.maxpool1
        self.Conv2d_3b_1x1 = base.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = base.Conv2d_4a_3x3
        self.maxpool2 = base.maxpool2
        self.Mixed_5b = base.Mixed_5b
        self.Mixed_5c = base.Mixed_5c
        self.Mixed_5d = base.Mixed_5d
        self.Mixed_6a = base.Mixed_6a
        self.Mixed_6b = base.Mixed_6b
        self.Mixed_6c = base.Mixed_6c
        self.Mixed_6d = base.Mixed_6d
        self.Mixed_6e = base.Mixed_6e
        self.Mixed_7a = base.Mixed_7a
        self.Mixed_7b = base.Mixed_7b
        self.Mixed_7c = base.Mixed_7c

    def forward(self, images):
        x = self.Conv2d_1a_3x3(images)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        low = self.Mixed_6e(x)
        x = self.Mixed_7a(low)
        x = self.Mixed_7b(x)
        high = self.Mixed_7c(x)
        return low, high

    @property
    def gradcam_target(self):
        return self.Mixed_7c


class ChannelSqueezeExcitation(nn.Module):
    """Lightweight channel attention used by the C-NMC SE-Inception model."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, features):
        weights = self.excitation(features.mean(dim=(-2, -1)))
        return features * weights[:, :, None, None]


# ==========================================
# 3. DYNAMIC GRAPH ATTENTION
# ==========================================

class DynamicGraphAttention(nn.Module):

    def __init__(
        self,
        dim=128,
        heads=4,
        k=6,
        dropout=0.2
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                "embed_dim must be divisible by heads"
            )

        self.heads = heads
        self.head_dim = dim // heads
        self.k = k

        self.qkv = nn.Linear(
            dim, 3 * dim, bias=False
        )

        self.output = nn.Linear(dim, dim)

        self.gate = nn.Parameter(
            torch.tensor(-1.0)
        )

        self.drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, nodes, spatial_mask):

        batch, count, dim = nodes.shape

        # Dynamic feature-based kNN graph
        with torch.no_grad():

            normalized = F.normalize(
                nodes.detach(),
                dim=-1
            )

            similarity = torch.bmm(
                normalized,
                normalized.transpose(1, 2)
            )

            diagonal = torch.eye(
                count,
                dtype=torch.bool,
                device=nodes.device
            )

            similarity.masked_fill_(
                diagonal.unsqueeze(0),
                float("-inf")
            )

            nearest = similarity.topk(
                min(self.k, count - 1),
                dim=-1
            ).indices

            # Combine spatial and feature edges
            edge_mask = (
                spatial_mask
                .to(nodes.device)
                .unsqueeze(0)
                .expand(batch, -1, -1)
                .clone()
            )

            edge_mask.scatter_(
                2, nearest, True
            )

        # Multi-head graph attention
        qkv = self.qkv(nodes).reshape(
            batch,
            count,
            3,
            self.heads,
            self.head_dim
        )

        query, key, value = qkv.unbind(dim=2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~edge_mask.unsqueeze(1),
            torch.finfo(scores.dtype).min
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.drop(attention)

        messages = torch.matmul(
            attention,
            value
        )

        messages = (
            messages
            .transpose(1, 2)
            .reshape(batch, count, dim)
        )

        # Residual graph update
        nodes = self.norm1(
            nodes
            + torch.sigmoid(self.gate)
            * self.drop(self.output(messages))
        )

        return self.norm2(
            nodes + self.ffn(nodes)
        )


# ==========================================
# 4. HYBRID CNN + GNN
# ==========================================

class HybridCNNGNN(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=False,
        freeze_backbone=False,
        embed_dim=128,
        gnn_layers=2,
        gnn_dropout=0.2,
        graph_neighbors=8,
        dropout=0.35,
        cnn_backbone_name="efficientnet_b3",
        channel_attention=False,
    ):
        super().__init__()

        if cnn_backbone_name not in {
            "efficientnet_b3", "efficientnet_v2_s", "inception_v3"
        }:
            raise ValueError(
                "Supported backbones: efficientnet_b3, efficientnet_v2_s, inception_v3"
            )

        if embed_dim % 4 != 0 or gnn_layers < 1:
            raise ValueError(
                "embed_dim must be divisible by 4 "
                "and gnn_layers >= 1"
            )

        if graph_neighbors not in (4, 8):
            raise ValueError(
                "graph_neighbors must be 4 or 8"
            )

        if cnn_backbone_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained_backbone else None
            base = efficientnet_b3(weights=weights)
            self.cnn_backbone = EfficientNetB3Features(base)
            low_channels, high_channels = 136, 1536
        elif cnn_backbone_name == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained_backbone else None
            base = efficientnet_v2_s(weights=weights)
            self.cnn_backbone = EfficientNetV2SFeatures(base)
            low_channels, high_channels = 160, 1280
        else:
            weights = Inception_V3_Weights.DEFAULT if pretrained_backbone else None
            base = inception_v3(weights=weights, init_weights=False)
            self.cnn_backbone = InceptionV3Features(base)
            low_channels, high_channels = 768, 2048

        self.cnn_backbone_name = cnn_backbone_name
        self.graph_neighbors = graph_neighbors
        self.channel_attention = channel_attention
        self.low_channel_attention = (
            ChannelSqueezeExcitation(low_channels) if channel_attention else nn.Identity()
        )
        self.high_channel_attention = (
            ChannelSqueezeExcitation(high_channels) if channel_attention else nn.Identity()
        )

        self.uses_pretrained_weights = (
            weights is not None
        )

        self.initialization_source = (
            f"ImageNet: {weights}"
            if weights is not None
            else f"random: {cnn_backbone_name}"
        )

        if freeze_backbone:
            for p in self.cnn_backbone.parameters():
                p.requires_grad_(False)

        # Multi-scale feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(
                low_channels + high_channels,
                embed_dim,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(
                8 if embed_dim % 8 == 0 else 4,
                embed_dim
            ),
            nn.GELU()
        )

        # Dynamic GNN layers
        self.graph_layers = nn.ModuleList([
            DynamicGraphAttention(
                dim=embed_dim,
                heads=4,
                k=6,
                dropout=gnn_dropout
            )
            for _ in range(gnn_layers)
        ])

        # Attention-based graph pooling
        self.node_score = nn.Linear(
            embed_dim, 1
        )

        self.graph_readout = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Linear(3 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CNN feature branch
        self.cnn_readout = nn.Sequential(
            nn.Linear(high_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # CNN + GNN feature fusion
        self.fusion_gate = nn.Linear(
            2 * embed_dim,
            embed_dim
        )

        nn.init.constant_(
            self.fusion_gate.bias,
            0.6
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            embed_dim,
            num_classes
        )

    # Preserve pretrained BatchNorm statistics
    def train(self, mode=True):

        super().train(mode)

        if mode and self.uses_pretrained_weights:
            for module in self.cnn_backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        return self

    def forward_features(self, images):

        # Multi-scale CNN features
        low, high = self.cnn_backbone(images)
        low = self.low_channel_attention(low)
        high = self.high_channel_attention(high)

        output_size = high.shape[-2:]
        # MPS cannot pool non-divisible grids (19x19 -> 10x10 at 300px).
        # Device copies preserve autograd and the exact adaptive-pooling result.
        if low.device.type == "mps" and any(
            source % target for source, target in zip(low.shape[-2:], output_size)
        ):
            low = F.adaptive_avg_pool2d(low.cpu(), output_size).to(high.device)
        else:
            low = F.adaptive_avg_pool2d(low, output_size)

        combined = torch.cat(
            [low, high],
            dim=1
        )

        feature_map = self.projection(combined)

        _, _, height, width = feature_map.shape

        # Spatial graph
        adjacency = grid_adjacency(
            height,
            width,
            self.graph_neighbors
        ).to(feature_map.device)

        # Feature maps into graph nodes
        nodes = feature_map.flatten(2).transpose(1, 2)

        # Dynamic graph attention
        for layer in self.graph_layers:
            nodes = layer(nodes, adjacency)

        # Graph-level pooling
        attention = torch.softmax(
            self.node_score(nodes),
            dim=1
        )

        weighted_pool = (
            attention * nodes
        ).sum(dim=1)

        pooled = torch.cat([
            nodes.mean(dim=1),
            nodes.amax(dim=1),
            weighted_pool
        ], dim=1)

        graph_features = self.graph_readout(pooled)

        # CNN representation
        cnn_features = self.cnn_readout(
            high.mean(dim=(-2, -1))
        )

        # Adaptive CNN-GNN fusion
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat(
                    [graph_features, cnn_features],
                    dim=1
                )
            )
        )

        features = (
            gate * graph_features
            + (1 - gate) * cnn_features
        )

        return features

    def forward_with_features(self, images):

        features = self.forward_features(images)

        logits = self.fc(
            self.dropout(features)
        )

        return logits, features

    @property
    def gradcam_target(self):
        return self.cnn_backbone.gradcam_target

    def forward(self, images):

        return self.forward_with_features(images)[0]


# ==========================================
# 5. MODEL FACTORY
# COMPATIBLE WITH run_grouped_experiment.py
# ==========================================

def create_model(
    num_classes,
    name="hybrid_cnn_gnn",
    pretrained_backbone=False,
    freeze_backbone=False,
    dropout=0.35,
    embed_dim=128,
    cnn_backbone_name="efficientnet_b3",
    gnn_layers=2,
    gnn_dropout=0.2,
    graph_neighbors=8,
    channel_attention=False,
):

    if name != "hybrid_cnn_gnn":
        raise ValueError(
            "Only the hybrid_cnn_gnn model is available"
        )

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
        channel_attention=channel_attention,
    )
