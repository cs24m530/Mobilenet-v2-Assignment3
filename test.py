#!/usr/bin/env python3
# test.py
# Single-run reproducer for the best sweep config (no training, one evaluation pass)

import argparse
import os
import time
import math
import copy
import struct
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# -----------------------
# Best config (from your sweep)
# -----------------------
BEST_CFG = {
    "act_bits": 8,
    "weight_bits": 8,
    "final_sparsity": 0.10312826477585516,
    "floor_frac": 0.008253317423732422,
    "lr": 0.020316716155312046,
    "seed": 42,
    "batch_size": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# -----------------------
# Minimal helpers (fallback if not importing from repo)
# -----------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_loaders(batch_size: int, num_workers: int = 4):
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
def load_qmod_into_model(qmod_path: str, model: nn.Module, device: str):
    """
    Read .qmod written by export_compressed_model and load quantized tensors back into model.
    This reconstructs int8 tensors and scales; we dequantize to float32 and load into matching params.
    """
    if not os.path.exists(qmod_path):
        raise FileNotFoundError(qmod_path)
    with open(qmod_path, "rb") as f:
        magic = f.read(4)
        if magic != b"QMOD":
            raise RuntimeError("Not a QMOD file")
        n_entries = struct.unpack("<I", f.read(4))[0]
        entries = {}
        for _ in range(n_entries):
            klen = struct.unpack("<H", f.read(2))[0]
            key = f.read(klen).decode("utf8")
            ndims = struct.unpack("<B", f.read(1))[0]
            dims = []
            for _ in range(ndims):
                dims.append(struct.unpack("<I", f.read(4))[0])
            nscales = struct.unpack("<I", f.read(4))[0]
            scales = [struct.unpack("<f", f.read(4))[0] for _ in range(nscales)]
            raw_len = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(raw_len)
            qt = torch.tensor(bytearray(raw), dtype=torch.int8)
            qt = qt.view(*dims)
            entries[key] = (qt, scales)
    # Map entries into model parameters (dequantize)
    sd = {}
    for key, (qt, scales) in entries.items():
        if key.endswith(".weight"):
            # per-channel if scales length == out_ch
            if len(scales) == qt.shape[0]:
                scale = torch.tensor(scales, dtype=torch.float32).view(-1,1,1,1)
                w = (qt.to(torch.float32) * scale).to(device)
            else:
                scale = float(scales[0])
                w = (qt.to(torch.float32) * scale).to(device)
            sd_key = key.replace(".weight", ".weight")
            sd[sd_key] = w
        elif key.endswith(".bias"):
            scale = float(scales[0])
            b = (qt.to(torch.float32) * scale).to(device)
            sd_key = key.replace(".bias", ".bias")
            sd[sd_key] = b
    # Load into model state_dict (non-strict)
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
    model.load_state_dict(model_sd, strict=False)
    return model

def make_mobilenet_v2(num_classes: int = 10):
    model = torchvision.models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# --- Fake-quant helpers (copy from training script) ---
class FakeQuantPerChannelSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        q = torch.round(x / scale).clamp(qmin, qmax)
        return q * scale
    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None

class FakeQuantPerTensorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        q = torch.round(x / scale).clamp(qmin, qmax)
        return q * scale
    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None

def calc_scale_symmetric(x: torch.Tensor, bits: int, floor_frac: float, per_channel: bool = False) -> Tuple[torch.Tensor, int, int]:
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax
    if per_channel:
        # weight layout [out, in, kH, kW]
        max_abs = x.detach().abs().amax(dim=(1,2,3), keepdim=True)
        mean_abs = x.detach().abs().mean(dim=(1,2,3), keepdim=True)
        guard = floor_frac * (mean_abs + 1e-8)
        scale = torch.maximum(max_abs / qmax, guard)
        return scale, qmin, qmax
    else:
        max_abs = x.detach().abs().max()
        mean_abs = x.detach().abs().mean()
        guard = floor_frac * (mean_abs + 1e-8)
        scale = torch.maximum(max_abs / qmax, guard)
        return scale, qmin, qmax

def fake_quant_weights_per_channel(w: torch.Tensor, bits: int, floor_frac: float):
    scale, qmin, qmax = calc_scale_symmetric(w, bits, floor_frac, per_channel=True)
    wq = FakeQuantPerChannelSTE.apply(w, scale, qmin, qmax)
    return wq, scale, (qmin, qmax)

def fake_quant_tensor_per_tensor(x: torch.Tensor, bits: int, floor_frac: float):
    scale, qmin, qmax = calc_scale_symmetric(x, bits, floor_frac, per_channel=False)
    xq = FakeQuantPerTensorSTE.apply(x, scale, qmin, qmax)
    return xq, scale, (qmin, qmax)


# QAT wrapper (must match your training code)
class QATConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, w_bits: int, a_bits: int, floor_frac: float, mask: torch.Tensor = None):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.bias_flag = conv.bias is not None

        self.weight = nn.Parameter(conv.weight.data.clone())
        if self.bias_flag:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.floor_frac = floor_frac

        self.register_buffer("mask", torch.ones_like(self.weight) if mask is None else mask.clone())

    def forward(self, x: torch.Tensor):
        # apply mask
        w = self.weight * self.mask
        # per-channel fake-quant for weights (same as training)
        wq, _, _ = fake_quant_weights_per_channel(w, self.w_bits, self.floor_frac)
        # conv with quantized weights
        x = F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # per-tensor fake-quant for activations (same as training)
        xq, _, _ = fake_quant_tensor_per_tensor(x, self.a_bits, self.floor_frac)
        return xq


def convert_model_for_qat(model: nn.Module, w_bits: int, a_bits: int, floor_frac: float, masks: Dict[str, torch.Tensor]):
    # Replace Conv2d modules with QATConv2d preserving module hierarchy
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            parent = model
            *path, last = name.split(".")
            try:
                for p in path:
                    parent = getattr(parent, p)
            except Exception:
                continue
            mask = masks.get(name, None) if masks is not None else None
            qat_conv = QATConv2d(module, w_bits=w_bits, a_bits=a_bits, floor_frac=floor_frac, mask=mask)
            setattr(parent, last, qat_conv)
    return model

def evaluate(model: nn.Module, loader, device: str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total

# -----------------------
# Small utilities for storage accounting (approx)
# -----------------------
def model_weight_bytes_fp32(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total

def model_weight_bytes_quant_and_meta(model: nn.Module, w_bits: int) -> Tuple[int, int]:
    total_bits = 0
    meta_bytes = 0
    for name, module in model.named_modules():
        # Count convs and QATConv2d (weights per-channel)
        if isinstance(module, (nn.Conv2d, QATConv2d)):
            if hasattr(module, "weight"):
                w = module.weight.detach()
                total_bits += int(w.numel()) * int(w_bits)
                # per-output-channel metadata (one float scale per out channel)
                if w.dim() >= 1:
                    out_ch = int(w.shape[0])
                    meta_bytes += out_ch * 4
            if hasattr(module, "bias") and module.bias is not None:
                total_bits += int(module.bias.numel()) * int(w_bits)
                meta_bytes += 4
        # Linear layers: per-tensor scale + optional bias
        elif isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                w = module.weight.detach()
                total_bits += int(w.numel()) * int(w_bits)
                meta_bytes += 4  # one scale per tensor
            if hasattr(module, "bias") and module.bias is not None:
                total_bits += int(module.bias.numel()) * int(w_bits)
                meta_bytes += 4
    quant_bytes = math.ceil(total_bits / 8)
    return quant_bytes, meta_bytes


# -----------------------
# Main single-run logic
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="Path to saved state_dict.pth (from your sweep best run).")
    parser.add_argument("--qmod", type=str, default=None, help="Optional: path to .qmod compressed file (not required).")
    parser.add_argument("--batch_size", type=int, default=BEST_CFG["batch_size"])
    args = parser.parse_args()

    device = BEST_CFG["device"]
    set_seed(BEST_CFG["seed"])

    # Data
    _, test_loader = get_cifar10_loaders(args.batch_size)

    # Build model and convert to QAT wrapper so keys align with saved state_dict
    model = make_mobilenet_v2(num_classes=10)
    masks = {}  # we don't need masks for evaluation; empty dict preserves API
    model = convert_model_for_qat(model, w_bits=BEST_CFG["weight_bits"], a_bits=BEST_CFG["act_bits"],
                                  floor_frac=BEST_CFG["floor_frac"], masks=masks)
    model = model.to(device)

    # Load state dict (best run)
    if not os.path.exists(args.state):
        raise FileNotFoundError(f"State dict not found: {args.state}")
    print("[Test] Loading state dict:", args.state)
    if args.qmod:
        print("[Test] Loading qmod and reconstructing weights:", args.qmod)
        # First load BN and other params from state_dict
        sd = torch.load(args.state, map_location=device, weights_only=True)
        if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model_state", "model_state_dict"]):
            for k in ["state_dict", "model_state", "model_state_dict"]:
                if k in sd:
                    sd = sd[k]
                    break
        clean = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print("[Test] load_state_dict missing keys:", missing)
        print("[Test] load_state_dict unexpected keys:", unexpected)

        # Then overwrite Conv/Linear weights with qmod
        model = load_qmod_into_model(args.qmod, model, device)

    else:
        sd = torch.load(args.state, map_location=device, weights_only=True)
        if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model_state", "model_state_dict"]):
            for k in ["state_dict", "model_state", "model_state_dict"]:
                if k in sd:
                    sd = sd[k]
                    break
        clean = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
    print("[Test] load_state_dict missing keys:", missing)
    print("[Test] load_state_dict unexpected keys:", unexpected)

    # Single evaluation pass (no training)
    acc = evaluate(model, test_loader, device)
    print(f"[Test] Single-run accuracy: {acc*100:.4f}%")

    # Storage accounting (approx)
    fp32_weight_b = model_weight_bytes_fp32(model)
    quant_weight_b, weight_meta_b = model_weight_bytes_quant_and_meta(model, BEST_CFG["weight_bits"])
    def mb(b): return round(b / 1024.0 / 1024.0, 4)
    print("\n=== Storage (approx) ===")
    print("weights_fp32_mb:", mb(fp32_weight_b))
    print("weights_quant_mb:", mb(quant_weight_b))
    print("weight_meta_mb:", mb(weight_meta_b))

    # Print the best-run summary (values you provided) for easy comparison
    print("\n=== Best-run reference values (for comparison) ===")
    for k, v in {
        "final_qat_acc": 0.8574,
        "baseline_acc": 0.835,
        "weight_bits": BEST_CFG["weight_bits"],
        "act_bits": BEST_CFG["act_bits"],
        "final_sparsity": BEST_CFG["final_sparsity"],
        "floor_frac": BEST_CFG["floor_frac"],
        "lr": BEST_CFG["lr"],
    }.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
