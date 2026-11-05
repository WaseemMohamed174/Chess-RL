"""Detect the machine: RAM, CPU cores, NVIDIA GPU, CUDA driver level."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

APP_OVERHEAD_GB = 0.8
RAM_USABLE_FRAC = 0.75
AVAIL_FRAC = 0.90
VRAM_USABLE_FRAC = 0.90

# CUDA builds PyTorch publishes wheels for (newest first). Drivers are backward compatible.
CUDA_TAGS = ("cu128", "cu126", "cu124", "cu121")


@dataclass
class Hardware:
    ram_total: float = 0.0
    ram_avail: float = 0.0
    cores: int = 1
    gpu_name: str | None = None
    vram_total: float = 0.0
    vram_free: float = 0.0
    gpu_available: bool = False
    cuda_tag: str | None = None
    cuda_version: str = ""
    gpu_note: str = ""

    @property
    def safe_ram(self) -> float:
        usable = min(self.ram_avail * AVAIL_FRAC, self.ram_total * RAM_USABLE_FRAC)
        return max(usable - APP_OVERHEAD_GB, 0.0)

    @property
    def safe_vram(self) -> float:
        return max(self.vram_free * VRAM_USABLE_FRAC - 0.2, 0.0)

    def to_dict(self) -> dict:
        d = {k: (round(v, 2) if isinstance(v, float) else v)
             for k, v in self.__dict__.items()}
        d["safe_ram"] = round(self.safe_ram, 2)
        d["safe_vram"] = round(self.safe_vram, 2)
        return d


def _psutil():
    try:
        import psutil
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "psutil"])
        import psutil
    return psutil


def _smi(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", *args], stderr=subprocess.DEVNULL,
                                       timeout=8, text=True)
    except Exception:
        return None


def _nvidia():
    out = _smi(["--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    best = None
    for line in out.strip().splitlines():
        try:
            name, tot, free = [p.strip() for p in line.split(",")]
            row = (name, float(tot) / 1024, float(free) / 1024)
        except ValueError:
            continue
        if best is None or row[2] > best[2]:
            best = row
    return best


def _cuda_tag():
    out = _smi([])
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    if not m:
        return None, ""
    ver = (int(m.group(1)), int(m.group(2)))
    for tag in CUDA_TAGS:
        if ver >= (int(tag[2:4]), int(tag[4:])):
            return tag, f"{ver[0]}.{ver[1]}"
    return None, f"{ver[0]}.{ver[1]}"


def detect_hardware() -> Hardware:
    vm = _psutil().virtual_memory()
    base = dict(ram_total=vm.total / 1024**3, ram_avail=vm.available / 1024**3,
                cores=os.cpu_count() or 1)
    gpu = _nvidia()
    if not gpu:
        return Hardware(**base, gpu_note="no NVIDIA GPU detected")
    name, vt, vf = gpu
    tag, drv = _cuda_tag()
    if tag:
        return Hardware(**base, gpu_name=name, vram_total=vt, vram_free=vf,
                        gpu_available=True, cuda_tag=tag, cuda_version=drv)
    return Hardware(**base, gpu_name=name, vram_total=vt, vram_free=vf,
                    gpu_note=f"driver supports CUDA {drv or '?'} (need 12.1+)",
                    cuda_version=drv)
