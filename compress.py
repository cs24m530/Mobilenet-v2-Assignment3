#!/usr/bin/env python3
# compress.py - configurable compression pipeline (quantization, pruning, or both)
# Drop-in replacement for your previous script. Controlled by config.yaml key:
# compression_method: quantization | pruning | quant_prune

import os
import argparse
import time
import yaml
import struct
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# -----------------------
# Helpers
# -----------------------
def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_loaders(batch_size, num_workers=4):
    normalize = T.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
    train_tf = T.Compose([T.RandomCrop(32,padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def make_mobilenet_v2(num_classes=10):
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# -----------------------
# Fake quant + QAT wrapper
# -----------------------
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

def calc_scale_symmetric(x, bits, floor_frac, per_channel=False):
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

def fake_quant_weights_per_channel(w, bits, floor_frac):
    scale, qmin, qmax = calc_scale_symmetric(w, bits, floor_frac, per_channel=True)
    wq = FakeQuantPerChannelSTE.apply(w, scale, qmin, qmax)
    return wq, scale, (qmin, qmax)

def fake_quant_tensor_per_tensor(x, bits, floor_frac):
    scale, qmin, qmax = calc_scale_symmetric(x, bits, floor_frac, per_channel=False)
    xq = FakeQuantPerTensorSTE.apply(x, scale, qmin, qmax)
    return xq, scale, (qmin, qmax)

class QATConv2d(nn.Module):
    def __init__(self, conv, w_bits, a_bits, floor_frac, mask=None):
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
        self.bias = nn.Parameter(conv.bias.data.clone()) if self.bias_flag else None

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.floor_frac = floor_frac

        self.register_buffer("mask", torch.ones_like(self.weight) if mask is None else mask.clone())

    def forward(self, x):
        w = self.weight * self.mask
        wq, _, _ = fake_quant_weights_per_channel(w, self.w_bits, self.floor_frac)
        x = F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        xq, _, _ = fake_quant_tensor_per_tensor(x, self.a_bits, self.floor_frac)
        return xq

def convert_model_for_qat(model, w_bits, a_bits, floor_frac, masks):
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

# -----------------------
# Pruning helpers (manual)
# -----------------------
def is_pointwise_conv(m):
    return isinstance(m, nn.Conv2d) and m.kernel_size == (1,1)

def magnitude_threshold(weight, target_sparsity):
    w_flat = weight.detach().abs().view(-1)
    k = int(target_sparsity * w_flat.numel())
    if k <= 0:
        return 0.0
    vals, _ = torch.topk(w_flat, k, largest=False)
    return float(vals.max().item())

def build_or_update_masks(model, masks, target_sparsity):
    new_masks = {}
    for name, module in model.named_modules():
        if is_pointwise_conv(module):
            W = module.weight.data
            thr = magnitude_threshold(W, target_sparsity)
            mask = (W.abs() > thr).float()
            if name in masks:
                mask = mask * masks[name]
            new_masks[name] = mask
    return new_masks

def apply_pruning_masks(model, masks):
    with torch.no_grad():
        for name, module in model.named_modules():
            if is_pointwise_conv(module) and name in masks:
                module.weight.mul_(masks[name])

def mask_optimizer_grads(model, masks):
    for name, module in model.named_modules():
        if is_pointwise_conv(module) and name in masks and module.weight.grad is not None:
            module.weight.grad.mul_(masks[name])

# -----------------------
# Storage accounting + export
# -----------------------
def model_weight_bytes_fp32(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total

def model_weight_bytes_quant_and_meta(model, w_bits):
    total_bits = 0
    meta_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, QATConv2d)):
            if hasattr(module, "weight"):
                w = module.weight.detach()
                total_bits += int(w.numel()) * int(w_bits)
                if w.dim() >= 1:
                    out_ch = int(w.shape[0])
                    meta_bytes += out_ch * 4
            if hasattr(module, "bias") and module.bias is not None:
                total_bits += int(module.bias.numel()) * int(w_bits)
                meta_bytes += 4
        elif isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                w = module.weight.detach()
                total_bits += int(w.numel()) * int(w_bits)
                meta_bytes += 4
            if hasattr(module, "bias") and module.bias is not None:
                total_bits += int(module.bias.numel()) * int(w_bits)
                meta_bytes += 4
    quant_bytes = math.ceil(total_bits / 8)
    return quant_bytes, meta_bytes

def export_compressed_model(model, w_bits, out_path):
    entries = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, QATConv2d)):
            if not hasattr(module, "weight"):
                continue
            w = module.weight.detach().cpu()
            scale, qmin, qmax = calc_scale_symmetric(w, w_bits, 0.01, per_channel=True)
            q = torch.clamp(torch.round(w / scale), qmin, qmax).to(torch.int8)
            entries.append((name + ".weight", q, scale.view(-1).cpu().numpy()))
            if hasattr(module, "bias") and module.bias is not None:
                b = module.bias.detach().cpu()
                bscale, _, _ = calc_scale_symmetric(b, w_bits, 0.01, per_channel=False)
                qb = torch.clamp(torch.round(b / bscale), -((2**(w_bits-1))-1), (2**(w_bits-1))-1).to(torch.int8)
                entries.append((name + ".bias", qb, [float(bscale.cpu().numpy())]))
        elif isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                w = module.weight.detach().cpu()
                scale, _, _ = calc_scale_symmetric(w, w_bits, 0.01, per_channel=False)
                q = torch.clamp(torch.round(w / scale), -((2**(w_bits-1))-1), (2**(w_bits-1))-1).to(torch.int8)
                entries.append((name + ".weight", q, [float(scale.cpu().numpy())]))
            if hasattr(module, "bias") and module.bias is not None:
                b = module.bias.detach().cpu()
                bscale, _, _ = calc_scale_symmetric(b, w_bits, 0.01, per_channel=False)
                qb = torch.clamp(torch.round(b / bscale), -((2**(w_bits-1))-1), (2**(w_bits-1))-1).to(torch.int8)
                entries.append((name + ".bias", qb, [float(bscale.cpu().numpy())]))
    with open(out_path, "wb") as f:
        f.write(b"QMOD")
        f.write(struct.pack("<I", len(entries)))
        for key, qtensor, scales in entries:
            kb = key.encode("utf8"); f.write(struct.pack("<H", len(kb))); f.write(kb)
            dims = qtensor.shape; f.write(struct.pack("<B", len(dims)))
            for d in dims: f.write(struct.pack("<I", int(d)))
            f.write(struct.pack("<I", len(scales)))
            for s in scales: f.write(struct.pack("<f", float(s)))
            raw = qtensor.numpy().tobytes(); f.write(struct.pack("<Q", len(raw))); f.write(raw)
    print("[Export] Compressed model written to:", out_path)

# -----------------------
# Training + evaluation
# -----------------------
def train_one_epoch(model, loader, optimizer, device, masks=None):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if masks:
            mask_optimizer_grads(model, masks)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item(); total += y.numel()
    return correct / total

def load_fp32_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ck, dict) and ("model_state_dict" in ck or "state_dict" in ck or "model_state" in ck):
        for k in ["model_state_dict", "state_dict", "model_state"]:
            if k in ck:
                return ck[k]
    return ck

# -----------------------
# Compression pipeline helpers
# -----------------------
def manual_prune_schedule(model, train_loader, cfg, device):
    """
    Apply iterative magnitude pruning during baseline training according to schedule in cfg.
    Returns final masks dictionary.
    """
    masks = {}
    optimizer = optim.SGD(model.parameters(), lr=cfg.get("lr_baseline", 0.1), momentum=0.9, weight_decay=5e-4)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get("lr_milestones", [15,25]), gamma=0.1)
    epochs = cfg.get("epochs_baseline", 30)
    for epoch in range(epochs):
        if epoch < cfg.get("prune_begin", 5):
            target_sparsity = 0.0
        elif epoch > cfg.get("prune_end", 20):
            target_sparsity = cfg.get("final_sparsity", 0.1)
        else:
            alpha = (epoch - cfg.get("prune_begin", 5)) / max(1, (cfg.get("prune_end", 20) - cfg.get("prune_begin", 5)))
            target_sparsity = alpha * cfg.get("final_sparsity", 0.1)
        masks = build_or_update_masks(model, masks, target_sparsity)
        apply_pruning_masks(model, masks)
        loss = train_one_epoch(model, train_loader, optimizer, device, masks)
        lr_sched.step()
        print(f"[Prune] Epoch {epoch} loss={loss:.4f} spars={target_sparsity:.4f}")
    return masks

def apply_quantization_only(model, cfg, masks):
    return convert_model_for_qat(model, cfg.get("weight_bits", 8), cfg.get("act_bits", 8), cfg.get("floor_frac", 0.01), masks or {})

def apply_pruning_only(model, cfg, train_loader, device):
    # run pruning schedule and keep model as pruned FP32
    masks = manual_prune_schedule(model, train_loader, cfg, device)
    apply_pruning_masks(model, masks)
    return model, masks

def apply_quant_prune(model, cfg, train_loader, device):
    masks = manual_prune_schedule(model, train_loader, cfg, device)
    apply_pruning_masks(model, masks)
    qat_model = copy.deepcopy(model)
    qat_model = convert_model_for_qat(qat_model, cfg.get("weight_bits", 8), cfg.get("act_bits", 8), cfg.get("floor_frac", 0.01), masks)
    return qat_model, masks

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--fp32", type=str, required=True, help="Path to baseline fp32 checkpoint")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--no_finetune", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    train_loader, test_loader = get_cifar10_loaders(cfg.get("batch_size", 128))

    # load fp32 checkpoint into model
    state = load_fp32_checkpoint(args.fp32, device)
    model = make_mobilenet_v2(num_classes=cfg.get("num_classes", 10)).to(device)
    model.load_state_dict(state, strict=False)
    baseline_acc = evaluate(model, test_loader, device)
    print("[Compress] Baseline acc:", baseline_acc)

    compression_method = cfg.get("compression_method", "quant_prune").lower()
    final_masks = {}

    if compression_method == "pruning":
        print("[Compress] Running pruning-only pipeline")
        model, final_masks = apply_pruning_only(model, cfg, train_loader, device)
        compressed_model = model  # pruned FP32
    elif compression_method == "quantization":
        print("[Compress] Running quantization-only pipeline")
        # no pruning schedule; convert directly to QAT wrapper (masks empty)
        compressed_model = apply_quantization_only(model, cfg, masks={})
    elif compression_method == "quant_prune":
        print("[Compress] Running pruning then quantization pipeline")
        compressed_model, final_masks = apply_quant_prune(model, cfg, train_loader, device)
    else:
        raise ValueError(f"Unknown compression_method: {compression_method}")

    # Optional QAT finetune if model is QAT wrapper and user didn't disable finetune
    if isinstance(compressed_model, nn.Module) and not args.no_finetune and compression_method in ("quantization", "quant_prune"):
        optimizer_qat = optim.SGD(compressed_model.parameters(), lr=cfg.get("lr_baseline", 0.1) * cfg.get("qat_lr_scale", 0.05), momentum=0.9, weight_decay=0.0)
        for e in range(cfg.get("qat_epochs", 20)):
            loss_q = train_one_epoch(compressed_model, train_loader, optimizer_qat, device, final_masks)
            acc_q = evaluate(compressed_model, test_loader, device)
            print(f"[QAT] {e+1}/{cfg.get('qat_epochs',20)} loss={loss_q:.4f} acc={acc_q*100:.2f}%")

    final_acc = evaluate(compressed_model, test_loader, device)
    print("[Final] Compressed model acc:", final_acc)

    # export compressed model and save final state
    ts = time.strftime("%Y%m%d_%H%M%S")
    wbits = cfg.get("weight_bits", 8)
    abits = cfg.get("act_bits", 8)
    spars = cfg.get("final_sparsity", 0.0)
    qmod_path = os.path.join(args.out_dir, f"mobilenetv2_compressed_w{wbits}_a{abits}_s{spars}_{ts}.qmod")
    export_compressed_model(compressed_model, wbits, qmod_path)
    state_path = qmod_path + ".state_dict.pth"
    torch.save(compressed_model.state_dict(), state_path)
    print("[Save] Compressed state saved:", state_path)

    # storage accounting
    fp32_weight_b = model_weight_bytes_fp32(compressed_model)
    quant_weight_b, weight_meta_b = model_weight_bytes_quant_and_meta(compressed_model, wbits)
    def mb(b): return round(b / 1024.0 / 1024.0, 4)
    print("\n=== Storage (approx) ===")
    print("weights_fp32_mb:", mb(fp32_weight_b))
    print("weights_quant_mb:", mb(quant_weight_b))
    print("weight_meta_mb:", mb(weight_meta_b))

    # optional wandb logging
    if args.use_wandb:
        try:
            import wandb
            run = wandb.init(project=cfg.get("wandb_project", "mobilenetv2_compress_sweep"), config=cfg, reinit=True)
            run.log({"final_qat_acc": final_acc})
            run.save(qmod_path, policy="now"); run.save(state_path, policy="now")
            artifact = wandb.Artifact(name=os.path.basename(qmod_path), type="compressed_model")
            artifact.add_file(qmod_path); artifact.add_file(state_path)
            run.log_artifact(artifact)
            run.finish()
        except Exception as ex:
            print("[WandB] optional logging failed:", ex)

if __name__ == "__main__":
    main()
