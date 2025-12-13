#!/usr/bin/env python3
# train.py - baseline training, saves only best fp32 checkpoint

import os, argparse, time, yaml
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

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

def evaluate(model, loader, device):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item(); total += y.numel()
    return correct/total

def train_one_epoch(model, loader, optimizer, device):
    model.train(); total_loss=0.0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))
    train_loader, test_loader = get_cifar10_loaders(cfg.get("batch_size",128))
    model = make_mobilenet_v2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.get("lr_baseline",0.1), momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(cfg.get("epochs_baseline",30)*0.5)], gamma=0.1)
    best_acc = 0.0; best_path = None
    for epoch in range(cfg.get("epochs_baseline",50)):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"[Baseline] Epoch {epoch} loss={loss:.4f} acc={acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.save_dir, "baseline_fp32_best.pth")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "acc": acc}, best_path)
    print("[Baseline] Best acc:", best_acc, "saved:", best_path)

if __name__ == "__main__":
    main()
