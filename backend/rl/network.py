"""Policy + value ResNet (matches the structure shown on the Architecture page).

    Input (18x8x8) -> Conv3x3+BN+ReLU -> N x ResidualBlock
        |- policy head: Conv1x1 -> BN -> ReLU -> Flatten -> FC(4672)
        '- value head : Conv1x1 -> BN -> ReLU -> FC(256) -> ReLU -> FC(1) -> Tanh

The constants must stay in sync with backend/cli/select.py (it estimates size
from them; tests/test_rl_core.py checks that the two agree)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_PLANES, POLICY_SIZE

POLICY_CONV = 32
VALUE_CONV = 1
VALUE_HIDDEN = 256


class ResidualBlock(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.conv1 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, blocks: int = 3, filters: int = 64):
        super().__init__()
        self.blocks, self.filters = blocks, filters
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResidualBlock(filters) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, POLICY_CONV, 1, bias=False), nn.BatchNorm2d(POLICY_CONV),
            nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(POLICY_CONV * 64, POLICY_SIZE))
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, VALUE_CONV, 1, bias=False), nn.BatchNorm2d(VALUE_CONV),
            nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(VALUE_CONV * 64, VALUE_HIDDEN), nn.ReLU(inplace=True),
            nn.Linear(VALUE_HIDDEN, 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(1)

    def config(self) -> dict:
        return {"blocks": self.blocks, "filters": self.filters}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_network(blocks: int, filters: int) -> PolicyValueNet:
    return PolicyValueNet(blocks, filters)


def layer_table(net: PolicyValueNet) -> list[dict]:
    """Rows for the Architecture page: name, type, output shape, parameters."""
    rows: list[dict] = []
    hooks = []

    def make_hook(name):
        def hook(mod, _inp, out):
            shape = list(out.shape[1:]) if torch.is_tensor(out) else None
            rows.append({"name": name, "type": type(mod).__name__, "output_shape": shape,
                         "params": sum(p.numel() for p in mod.parameters(recurse=False))})
        return hook

    for name, mod in net.named_modules():
        if name and not list(mod.children()):         # leaf layers only
            hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, INPUT_PLANES, 8, 8, device=next(net.parameters()).device))
    net.train(was_training)
    for h in hooks:
        h.remove()
    return rows
