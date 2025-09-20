# =============================== #
# Fast CIFAR-10 Trainer (Local)
# - Transfer learning: ResNet18 (frozen backbone)
# - AMP, channels_last, fast dataloaders
# - No internet download (local data only)
# - Optional VERY FAST mode for smoke test
# =============================== #

import os, time, argparse, random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ---------------- Utils ---------------- #
def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Faster matmul on Ampere+ (PyTorch 2.x)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------------- Data ---------------- #
def make_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    very_fast: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    try:
        trainset_full = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_train)
        testset       = datasets.CIFAR10(root=data_root, train=False, download=False, transform=tf_eval)
    except RuntimeError as e:
        raise RuntimeError(
            "\n[CIFAR-10 NOT FOUND]\n"
            f"Expected local data under: {data_root}\n"
            "Put either the extracted folder 'cifar-10-batches-py' or the archive "
            "'cifar-10-python.tar.gz' inside that folder (download=False).\n"
            f"Original error: {e}"
        )

    # quick split: 45k train / 5k val
    n_total = len(trainset_full)
    n_val = 5000
    n_train = n_total - n_val
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # VERY FAST mode: use only a small subset (e.g., 10k train, 1k val, 1k test)
    if very_fast:
        trainset = Subset(trainset, list(range(min(10000, len(trainset)))))
        valset   = Subset(valset,   list(range(min(1000, len(valset)))))
        testset  = Subset(testset,  list(range(min(1000, len(testset)))))
    # else keep full testset

    # DataLoader performance knobs
    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader, test_loader

# ------------- Model (Fast) ------------- #
def make_fast_transfer(num_classes: int = 10, freeze_backbone: bool = True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    if freeze_backbone:
        for n, p in m.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return m

# ------------- Train/Eval ------------- #
def train_one_epoch(model, loader, optimizer, device, scaler=None, channels_last=False):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, channels_last=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return loss_sum/total, correct/total

def fit_fast(model, train_loader, val_loader, device, epochs=3, lr=3e-3, wd=1e-4):
    # Only a few params (fc layer) -> higher LR works well
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    # Cosine schedule without restarts (cheap & good)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_state = None
    best_val_acc = -1.0
    channels_last = (device.type == "cuda")

    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, channels_last)
        val_loss, val_acc = evaluate(model, val_loader, device, channels_last)
        scheduler.step()

        print(f"[Fast] Epoch {ep:02d}/{epochs} | "
              f"Train {tr_acc*100:.2f}% / {tr_loss:.4f} | "
              f"Val {val_acc*100:.2f}% / {val_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ------------- Main ------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"C:\Users\Dell\Desktop\ACV\cifar10data")
    parser.add_argument("--batch_size", type=int, default=256)     # larger batch for speed (fits on most GPUs)
    parser.add_argument("--epochs", type=int, default=3)           # quick but good results
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4)//2))
    parser.add_argument("--very_fast", action="store_true", help="train on a tiny subset for a ~seconds run")
    parser.add_argument("--out", type=str, default="outputs_fast")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        args.data_root, args.batch_size, args.num_workers, very_fast=args.very_fast
    )

    model = make_fast_transfer(num_classes=10, freeze_backbone=True).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    t0 = time.time()
    model = fit_fast(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, wd=args.wd)
    test_loss, test_acc = evaluate(model, test_loader, device, channels_last=(device.type=="cuda"))
    dt = time.time() - t0

    torch.save(model.state_dict(), os.path.join(args.out, "resnet18_fast_cifar10.pt"))
    print(f"\nDone in {dt:.1f}s | Test Acc: {test_acc*100:.2f}% | Saved -> {os.path.join(args.out, 'resnet18_fast_cifar10.pt')}")
    print(f"Trainable params (M): {count_params(model)/1e6:.3f}")

if __name__ == "__main__":
    main()
