"""One learner: samples batches from the replay buffer and updates the network.

    loss = (z - v)^2  -  pi . log p        (+ weight decay in the optimizer)

Mixed precision on CUDA (bfloat16 when the GPU supports it, else float16 with a
loss scaler), channels_last memory layout, cudnn autotune."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig


class Trainer:
    def __init__(self, net, cfg: TrainConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.net = net.to(self.device)
        if self.cuda:
            torch.backends.cudnn.benchmark = True
            self.net = self.net.to(memory_format=torch.channels_last)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        bf16 = self.cuda and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.cuda and not bf16 else None
        self.step_count = 0

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def steps_for(self, buffer_len: int) -> int:
        c = self.cfg
        return int(max(1, min(c.max_steps_per_iteration,
                              c.epochs_per_iteration * buffer_len // c.batch_size)))

    def train(self, buffer, steps: int, rng: np.random.Generator, should_stop=None) -> dict:
        self.net.train()
        pl = vl = 0.0
        done = 0
        for _ in range(steps):
            if should_stop and should_stop():
                break
            x, pi, z = buffer.sample(self.cfg.batch_size, rng)
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
            if self.cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            pi = torch.from_numpy(pi).to(self.device, non_blocking=True)
            z = torch.from_numpy(z).to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.cuda):
                logits, v = self.net(x)
            p_loss = -(pi * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(v.float(), z)
            loss = p_loss + v_loss
            self.opt.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                self.opt.step()
            pl += float(p_loss)
            vl += float(v_loss)
            done += 1
            self.step_count += 1
        self.net.eval()
        n = max(done, 1)
        return {"policy_loss": pl / n, "value_loss": vl / n, "steps": done}

    def save(self, path) -> None:
        torch.save({"config": self.net.config(), "model": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "steps": self.step_count}, path)

    def load_optimizer(self, path) -> None:
        ck = torch.load(path, map_location=self.device)
        if ck.get("opt"):
            self.opt.load_state_dict(ck["opt"])
        self.step_count = ck.get("steps", 0)


def load_network(path, device: str = "cpu"):
    """Rebuild a network from a checkpoint file written by Trainer.save()."""
    from .network import build_network
    ck = torch.load(path, map_location=device)
    net = build_network(**ck["config"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval()
