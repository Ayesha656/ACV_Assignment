"""
Task 4: CNN with Regularization and Hyperparameter Tuning (PyTorch, CIFAR-10)

What this script does
- Builds a CNN: [Conv → BN → ReLU → MaxPool] x 3 → FC + Dropout → 10 logits
- Trains for ≥20 epochs with Early Stopping (set FAST_MODE=False for full run)
- Systematic hyperparameter tuning:
    lr ∈ {1e-4, 1e-3, 1e-2}
    optimizer ∈ {SGD, Adam}
    dropout ∈ {0.2, 0.5}
    batch size ∈ {32, 64, 128}
- Runs 2 regimes:
    1) NO_AUG_REG  (no augmentation, no weight decay)
    2) WITH_AUG_REG (augmentation + weight decay)
- Saves:
    ./figs/best_learning_curves.png
    ./figs/confusion_matrix.png
    ./outputs/trials_summary_NO_AUG_REG.csv
    ./outputs/trials_summary_WITH_AUG_REG.csv
- Prints best hyperparameters + test accuracy

Install deps:
    pip install torch torchvision scikit-learn matplotlib numpy pandas
"""

import os, time, copy, random, numpy as np, matplotlib
matplotlib.use("Agg")  # save plots (no GUI)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# -----------------------------
# Repro / Paths / Config
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./cifar10data"
FIG_DIR = "./figs"; os.makedirs(FIG_DIR, exist_ok=True)
OUT_DIR = "./outputs"; os.makedirs(OUT_DIR, exist_ok=True)

# Speed knob for quick dry-runs (set False for final ≥20 epochs runs)
FAST_MODE = False   # True -> quick demo; False -> ≥20 epochs
EPOCHS   = 25 if not FAST_MODE else 8
PATIENCE = 5  if not FAST_MODE else 2
VAL_SIZE = 5000  # from 50k train

LR_GRID     = [1e-4, 1e-3, 1e-2]
OPTIM_GRID  = ["sgd", "adam"]
DROPOUT_GRID= [0.2, 0.5]
BS_GRID     = [32, 64, 128]
WD_WITH_REG = 5e-4   # weight decay in WITH_AUG_REG
WD_NO_REG   = 0.0    # weight decay in NO_AUG_REG

# -----------------------------
# Data transforms
# -----------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def make_transforms(augment: bool):
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, test_tf

# -----------------------------
# CNN model: Conv-BN-ReLU-MaxPool stacks → FC+Dropout → 10 logits
# -----------------------------
class CifarCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10)  # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits (use CrossEntropyLoss)

# -----------------------------
# Train / eval utilities
# -----------------------------
def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss, epoch_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        acc = accuracy(logits, yb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = yb.size(0)
        epoch_loss += loss.item() * bs
        epoch_acc  += acc * bs
        n += bs
    return epoch_loss / n, epoch_acc / n

def train_with_early_stopping(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler=None, patience=5):
    best_w = copy.deepcopy(model.state_dict())
    best_val = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    wait = 0

    for ep in range(1, epochs+1):
        tl, ta = run_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_one_epoch(model, val_loader,   criterion, optimizer=None)

        # IMPORTANT: pass validation loss to ReduceLROnPlateau (no verbose kwarg)
        if scheduler is not None:
            scheduler.step(vl)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(f"  Epoch {ep:02d}/{epochs} | train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")

        # Early stopping on best val accuracy
        if va > best_val:
            best_val = va
            best_w = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep} (best val acc={best_val:.4f}).")
                break

    model.load_state_dict(best_w)
    return model, history, best_val

def make_loaders(batch_size, augment):
    train_tf, test_tf = make_transforms(augment)
    train_full = datasets.CIFAR10(DATA_ROOT, train=True,  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)

    train_size = len(train_full) - VAL_SIZE
    val_size = VAL_SIZE
    train_set, val_set = random_split(
        train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    pin = True if device.type == "cuda" else False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, val_loader, test_loader

def build_optimizer(name, model, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(name)

# -----------------------------
# Grid search runner
# -----------------------------
def run_experiments(augment: bool, use_regularization: bool):
    regime = "WITH_AUG_REG" if augment and use_regularization else \
             "NO_AUG_REG"   if (not augment and not use_regularization) else \
             "MIXED"

    print(f"\n=== Running regime: {regime} ===")
    results = []
    best = {"val_acc": -1.0}

    for bs in BS_GRID:
        train_loader, val_loader, test_loader = make_loaders(bs, augment=augment)

        for dr in DROPOUT_GRID:
            for opt_name in OPTIM_GRID:
                for lr in LR_GRID:
                    # Model + optimizer
                    model = CifarCNN(dropout=dr).to(device)
                    wd = WD_WITH_REG if use_regularization else WD_NO_REG
                    optimizer = build_optimizer(opt_name, model, lr, weight_decay=wd)
                    criterion = nn.CrossEntropyLoss()

                    # Version-agnostic ReduceLROnPlateau (no 'verbose' kwarg)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=2
                    )

                    print(f"\n[Trial] bs={bs} dropout={dr} opt={opt_name} lr={lr} "
                          f"augment={augment} weight_decay={wd}")
                    t0 = time.time()
                    model, hist, best_val = train_with_early_stopping(
                        model, train_loader, val_loader, EPOCHS, optimizer, criterion,
                        scheduler=scheduler, patience=PATIENCE
                    )
                    train_time = time.time() - t0

                    # Evaluate on val & test
                    val_loss, val_acc = run_one_epoch(model, val_loader, criterion)
                    test_loss, test_acc = run_one_epoch(model, test_loader, criterion)

                    row = dict(regime=regime, bs=bs, dropout=dr, opt=opt_name, lr=lr,
                               weight_decay=wd, val_acc=val_acc, test_acc=test_acc,
                               val_loss=val_loss, test_loss=test_loss, train_time=train_time)
                    results.append(row)

                    # Track best (by val_acc)
                    if val_acc > best["val_acc"]:
                        best = {**row, "model_state": copy.deepcopy(model.state_dict()),
                                "history": hist,
                                "classes": datasets.CIFAR10(DATA_ROOT, train=False, download=True).classes}

    # Save trials table
    df = pd.DataFrame(results).sort_values(by="val_acc", ascending=False)
    csv_path = os.path.join(OUT_DIR, f"trials_summary_{regime}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    return best, df

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_learning_curves(history, out_path):
    epochs = range(1, len(history["train_acc"])+1)
    plt.figure(figsize=(10,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy vs epochs")
    plt.legend(); plt.grid(alpha=0.3)
    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs epochs")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[Saved] {out_path}")

def save_confusion_matrix(model, loader, class_names, out_path):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[Saved] {out_path}")

# -----------------------------
# Main: run NO_AUG_REG and WITH_AUG_REG, compare best
# -----------------------------
def main():
    t_all = time.time()

    # Regime A: no augmentation, no weight decay
    best_no, df_no = run_experiments(augment=False, use_regularization=False)

    # Regime B: with augmentation + weight decay
    best_yes, df_yes = run_experiments(augment=True, use_regularization=True)

    # Choose best overall by val acc
    best = best_yes if best_yes["val_acc"] >= best_no["val_acc"] else best_no
    print("\n=== Best Overall (by validation accuracy) ===")
    print({k: best[k] for k in ["regime","bs","dropout","opt","lr","weight_decay","val_acc","test_acc","train_time"]})

    # Rebuild loaders and model to produce final plots & confusion matrix
    train_loader, val_loader, test_loader = make_loaders(best["bs"], augment=(best["regime"]=="WITH_AUG_REG"))
    model = CifarCNN(dropout=best["dropout"]).to(device)
    model.load_state_dict(best["model_state"])

    # Learning curves
    lc_path = os.path.join(FIG_DIR, "best_learning_curves.png")
    plot_learning_curves(best["history"], lc_path)

    # Confusion matrix (test)
    class_names = datasets.CIFAR10(DATA_ROOT, train=False, download=True).classes
    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    save_confusion_matrix(model, test_loader, class_names, cm_path)

    # Final summary
    print("\n=== Summary ===")
    print(f"Best regime: {best['regime']}")
    print(f"Best hyperparameters: batch_size={best['bs']}, dropout={best['dropout']}, "
          f"optimizer={best['opt']}, lr={best['lr']}, weight_decay={best['weight_decay']}")
    print(f"Validation accuracy: {best['val_acc']:.4f}")
    print(f"Test accuracy: {best['test_acc']:.4f}")
    print(f"Training time (best run): {best['train_time']:.1f}s")
    print(f"Trials CSVs saved in: {OUT_DIR}")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"[Timing] Total wall time: {time.time()-t_all:.1f}s")

if __name__ == "__main__":
    main()
