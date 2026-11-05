"""Describe the network WITHOUT PyTorch (the gateway has none).

Layer names match PolicyValueNet.named_modules(); totals are checked against the
real network in tests/test_rl_core.py when PyTorch is available."""
from __future__ import annotations

INPUT_PLANES, POLICY_SIZE = 18, 4672
POLICY_CONV, VALUE_CONV, VALUE_HIDDEN = 32, 1, 256


def architecture(blocks: int, filters: int) -> dict:
    f = filters
    rows: list[dict] = []

    def add(name, kind, shape, params=0, macs=0, group=""):
        rows.append({"name": name, "type": kind, "output_shape": shape, "params": params,
                     "macs": macs, "group": group})

    def conv(name, cin, cout, k, group):
        add(name, "Conv2d", [cout, 8, 8], cin * k * k * cout, cin * k * k * cout * 64, group)

    def bn(name, c, group):
        add(name, "BatchNorm2d", [c, 8, 8], 2 * c, 0, group)

    def fc(name, i, o, group):
        add(name, "Linear", [o], i * o + o, i * o, group)

    conv("stem.0", INPUT_PLANES, f, 3, "Stem")
    bn("stem.1", f, "Stem")
    add("stem.2", "ReLU", [f, 8, 8], group="Stem")
    for b in range(blocks):
        g = f"Residual block {b + 1}"
        conv(f"tower.{b}.conv1", f, f, 3, g)
        bn(f"tower.{b}.bn1", f, g)
        conv(f"tower.{b}.conv2", f, f, 3, g)
        bn(f"tower.{b}.bn2", f, g)
    g = "Policy head"
    conv("policy_head.0", f, POLICY_CONV, 1, g)
    bn("policy_head.1", POLICY_CONV, g)
    add("policy_head.2", "ReLU", [POLICY_CONV, 8, 8], group=g)
    add("policy_head.3", "Flatten", [POLICY_CONV * 64], group=g)
    fc("policy_head.4", POLICY_CONV * 64, POLICY_SIZE, g)
    g = "Value head"
    conv("value_head.0", f, VALUE_CONV, 1, g)
    bn("value_head.1", VALUE_CONV, g)
    add("value_head.2", "ReLU", [VALUE_CONV, 8, 8], group=g)
    add("value_head.3", "Flatten", [VALUE_CONV * 64], group=g)
    fc("value_head.4", VALUE_CONV * 64, VALUE_HIDDEN, g)
    add("value_head.5", "ReLU", [VALUE_HIDDEN], group=g)
    fc("value_head.6", VALUE_HIDDEN, 1, g)
    add("value_head.7", "Tanh", [1], group=g)
    params = sum(r["params"] for r in rows)
    return {"blocks": blocks, "filters": filters, "input_shape": [INPUT_PLANES, 8, 8],
            "policy_size": POLICY_SIZE, "params": params,
            "size_mb": round(params * 4 / 1024 ** 2, 2),
            "macs": sum(r["macs"] for r in rows), "layers": rows}
