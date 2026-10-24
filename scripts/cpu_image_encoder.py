"""Frozen CPU ResNet18 image encoder, without a torchvision runtime dependency.

The network architecture is adapted from torchvision v0.20.1 models/resnet.py,
Copyright (c) Soumith Chintala 2016, under the BSD 3-Clause licence; see
docs/third_party_resnet18.txt. No random network
is offered as an extraction option: load_encoder requires the pinned official
ImageNet checkpoint, verifies its bytes and strictly loads every parameter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from extraction_contract import sha256

SCHEMA = "cpu-resnet18-six-camera-v1"
ENCODER = "resnet18-imagenet1k-v1"
HOOK = "resnet18 avgpool; six-camera concatenation"
WEIGHTS_NAME = "resnet18-f37072fd.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
LAYERS = [1]
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
VIEW_FEATURE_DIM = 512
FEATURE_DIM = 6 * VIEW_FEATURE_DIM


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Torchvision's standard two-convolution ResNet18 residual block."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """State-dict-compatible standard ResNet18 returning its pooled features."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Retain fc for strict validation of the complete official checkpoint;
        # its classification output is not used as the driving representation.
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)


def verify_weights(weights: Path) -> str:
    weights = Path(weights)
    if not weights.is_file() or weights.is_symlink():
        raise ValueError("A regular pinned official ResNet18 weights file is required")
    if len(WEIGHTS_SHA256) != 64 or sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("ResNet18 checkpoint SHA256 mismatch; "
                         "no random/unverified weights allowed")
    return WEIGHTS_SHA256


def load_encoder(weights: Path) -> ResNet18:
    verify_weights(weights)
    model = ResNet18()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Official ImageNet1K V1 resize/crop/normalise procedure, using PIL bilinear."""
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Image dimensions must be positive")
    if width <= height:
        size = (RESIZE_SHORT_SIDE, int(RESIZE_SHORT_SIDE * height / width))
    else:
        size = (int(RESIZE_SHORT_SIDE * width / height), RESIZE_SHORT_SIDE)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = (round((side - CROP_SIZE) / 2) for side in image.size)
    image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1)
    mean = value.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = value.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (value - mean) / std


def encode_views(model: nn.Module, paths: list[Path]) -> np.ndarray:
    if len(paths) != 6:
        raise ValueError("Exactly six ordered camera paths are required")
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(preprocess(image))
    with torch.inference_mode():
        encoded = model(torch.stack(tensors))
    if encoded.shape != (6, VIEW_FEATURE_DIM) or not torch.isfinite(encoded).all():
        raise ValueError("Encoder must produce six finite 512-dimensional camera representations")
    return encoded.reshape(FEATURE_DIM).cpu().numpy().astype(np.float32, copy=False)
