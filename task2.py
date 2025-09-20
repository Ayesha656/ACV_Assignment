"""
Task 2: Custom kNN + PCA + Hyperparameter Tuning on CIFAR-10
- From-scratch kNN (no sklearn KNeighborsClassifier)
- Feature sets: RAW pixels (flattened) and HOG descriptors
- PCA dims: 100 / 200 / 300
- Distances: Euclidean, Manhattan, Cosine
- K values: {1, 3, 5, 7, 11, 21}
- 3-fold cross-validation, select best hyperparams, evaluate on test
- Save "val_acc_vs_k_RAW.png" and "val_acc_vs_k_HOG.png"
"""

import os, random, numpy as np, matplotlib
matplotlib.use("Agg")  # save plots to files
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# ----------------------------
# Config / Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = "./cifar10data"
FIGS = "./figs"; os.makedirs(FIGS, exist_ok=True)

# Smaller subsets to keep runtime reasonable on CPU
N_TRAIN = 6000     # from 45k
N_VAL   = 1500     # from 5k
N_TEST  = 2000     # from 10k

# HOG params
HOG_KW = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False, block_norm="L2-Hys")

# Hyperparameter grids
K_VALUES = [1, 3, 5, 7, 11, 21]
DISTANCES = ["euclidean", "manhattan", "cosine"]
PCA_DIM = [100, 200, 300]

# ----------------------------
# Data Loading
# ----------------------------
print("[Info] Loading CIFAR-10...")

# RAW pipeline: only ToTensor() -> [0,1], NO per-channel normalization (keep raw-ish)
transform_raw = transforms.ToTensor()

# Base pipeline for HOG: ToTensor() only (we will convert to gray for HOG)
transform_base = transforms.ToTensor()

train_full = datasets.CIFAR10(root=ROOT, train=True,  download=True, transform=transform_base)
test_set   = datasets.CIFAR10(root=ROOT, train=False, download=True, transform=transform_base)
classes    = train_full.classes

# Split 45k/5k then subselect manageable sizes
train_set, val_set = random_split(
    train_full, [45000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

# Helper to take a random subset from a (Subset or Dataset)
def take_subset(ds, n):
    idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)
    # fetch tensors and labels
    X, y = [], []
    for i in idxs:
        img_t, lab = ds[i]
        X.append(img_t.numpy())  # C,H,W in [0,1]
        y.append(lab)
    return np.array(X), np.array(y)

print("[Info] Sampling subsets...")
Xtr_img, ytr = take_subset(train_set, N_TRAIN)
Xval_img, yval = take_subset(val_set, N_VAL)
Xte_img, yte = take_subset(test_set, N_TEST)

print(f"[Info] Shapes (images): train={Xtr_img.shape}, val={Xval_img.shape}, test={Xte_img.shape}")

# ----------------------------
# Feature Builders
# ----------------------------
def to_raw_flattened(X_img):
    """X_img: (N, C, H, W) in [0,1] -> (N, C*H*W) float32"""
    N = X_img.shape[0]
    return X_img.reshape(N, -1).astype(np.float32)

def to_hog_features(X_img):
    """Compute HOG for each image. Convert CHW->HWC->[0,1] gray, then hog."""
    feats = []
    for img in X_img:
        img_hwc = np.transpose(img, (1,2,0))  # CHW->HWC
        gray = rgb2gray(img_hwc)              # [0,1], HxW
        feat = hog(gray, **{**HOG_KW, "visualize": False})
        feats.append(feat.astype(np.float32))
    return np.vstack(feats)

# Build both feature sets
print("[Info] Building RAW and HOG features...")
Xtr_raw = to_raw_flattened(Xtr_img)
Xval_raw = to_raw_flattened(Xval_img)
Xte_raw = to_raw_flattened(Xte_img)

Xtr_hog = to_hog_features(Xtr_img)
Xval_hog = to_hog_features(Xval_img)
Xte_hog = to_hog_features(Xte_img)

print(f"[Info] RAW dims: train={Xtr_raw.shape}, HOG dims: train={Xtr_hog.shape}")

# ----------------------------
# Custom kNN (NumPy)
# ----------------------------
class KNNFromScratch:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)

    def _pairwise_dist(self, A, B):
        """
        Compute distances between A (nxd) and B (mxd) according to self.metric.
        Returns (n, m) matrix.
        """
        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            A2 = np.sum(A*A, axis=1, keepdims=True)
            B2 = np.sum(B*B, axis=1, keepdims=True).T
            d2 = A2 + B2 - 2*np.dot(A, B.T)
            d2 = np.maximum(d2, 0.0)
            return np.sqrt(d2, dtype=np.float32)
        elif self.metric == "manhattan":
            # expand dims and broadcast (may be slower but simple)
            # For memory, tile in chunks if needed; here sizes are modest.
            n, m = A.shape[0], B.shape[0]
            D = np.empty((n, m), dtype=np.float32)
            for i in range(n):
                D[i] = np.sum(np.abs(B - A[i]), axis=1)
            return D
        elif self.metric == "cosine":
            # cosine distance = 1 - cosine similarity
            A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(A_norm, B_norm.T)
            return (1.0 - sim).astype(np.float32)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X):
        X = X.astype(np.float32)
        D = self._pairwise_dist(X, self.X)  # (n_test, n_train)
        # find indices of k smallest distances
        idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]  # (n_test, k)
        # vote
        preds = []
        for row in idx:
            votes = self.y[row]
            # majority vote; tie-break by smallest label
            vals, counts = np.unique(votes, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds, dtype=np.int32)

# ----------------------------
# PCA + Standardization helpers
# ----------------------------
def fit_transform_pca(X_train, X_val, X_test, n_components):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X_train)
    Xval_s = scaler.transform(X_val)
    Xte_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=SEED)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xval_p = pca.transform(Xval_s)
    Xte_p = pca.transform(Xte_s)
    return (Xtr_p, Xval_p, Xte_p), scaler, pca

# ----------------------------
# Cross-Validation routine
# ----------------------------
def cross_val_grid(X, y, k_values, distances, pca_dims, nfolds=3, feature_name="RAW", plot_path=None):
    """
    Returns:
      best_params, best_val_acc, history (list of dict rows), per-k plot saved.
    """
    print(f"\n[CV] Feature: {feature_name}")
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=SEED)
    history = []

    # We'll also collect mean val acc vs k for each distance at the BEST PCA dim
    per_metric_k_acc = {metric: {k: [] for k in k_values} for metric in distances}
    best_overall = {"acc": -1.0, "k": None, "metric": None, "pca": None}

    for p in pca_dims:
        # Standardize + PCA once (fit on whole training set for CV folds' transforms),
        # then split folds on the transformed space (common in fast student setups).
        (X_p, _, _), scaler_global, pca_global = fit_transform_pca(X, X, X, n_components=p)

        for metric in distances:
            for K in k_values:
                fold_acc = []
                for tr_idx, va_idx in kf.split(X_p):
                    Xtr, Xva = X_p[tr_idx], X_p[va_idx]
                    ytr, yva = y[tr_idx], y[va_idx]

                    knn = KNNFromScratch(k=K, metric=metric)
                    knn.fit(Xtr, ytr)
                    pred = knn.predict(Xva)
                    acc = accuracy_score(yva, pred)
                    fold_acc.append(acc)

                mean_acc = float(np.mean(fold_acc))
                history.append({
                    "feature": feature_name, "pca": p, "metric": metric, "k": K, "cv_acc": mean_acc
                })

                # Track best overall
                if mean_acc > best_overall["acc"]:
                    best_overall = {"acc": mean_acc, "k": K, "metric": metric, "pca": p}

    # For plotting: choose the best PCA (argmax over avg across k/metric)
    # Compute metric-wise mean acc per k for the best PCA only
    best_pca = best_overall["pca"]
    rows_best_p = [r for r in history if r["pca"] == best_pca]
    # Fill per_metric_k_acc
    for r in rows_best_p:
        per_metric_k_acc[r["metric"]][r["k"]].append(r["cv_acc"])
    # Average duplicates (should be 1 here, but safe)
    avg_per_metric_k = {m: [np.mean(per_metric_k_acc[m][k]) for k in k_values] for m in distances}

    # Plot val acc vs k for each metric (at best_pca)
    if plot_path:
        plt.figure(figsize=(7,5))
        for metric in distances:
            plt.plot(k_values, avg_per_metric_k[metric], marker="o", label=f"{metric}")
        plt.title(f"Validation Accuracy vs k ({feature_name}, PCA={best_pca})")
        plt.xlabel("k"); plt.ylabel("CV Accuracy"); plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Saved] {plot_path}")

    return best_overall, history

# ----------------------------
# Run CV for RAW and HOG
# ----------------------------
# RAW: scale+PCA will be applied inside cross_val_grid
best_raw, hist_raw = cross_val_grid(
    Xtr_raw, ytr, K_VALUES, DISTANCES, PCA_DIM, nfolds=3,
    feature_name="RAW", plot_path=os.path.join(FIGS, "val_acc_vs_k_RAW.png")
)

# HOG:
best_hog, hist_hog = cross_val_grid(
    Xtr_hog, ytr, K_VALUES, DISTANCES, PCA_DIM, nfolds=3,
    feature_name="HOG", plot_path=os.path.join(FIGS, "val_acc_vs_k_HOG.png")
)

def print_best(name, best):
    print(f"\n[Best {name}]")
    print(f"  PCA components : {best['pca']}")
    print(f"  Distance       : {best['metric']}")
    print(f"  k              : {best['k']}")
    print(f"  CV Accuracy    : {best['acc']:.4f}")

print_best("RAW", best_raw)
print_best("HOG", best_hog)

# ----------------------------
# Fit on full train (with best params) and evaluate on held-out TEST
# ----------------------------
def final_train_and_test(X_train, y_train, X_test, y_test, best):
    # Standardize + PCA with best p
    (Xtr_p, _, Xte_p), scaler, pca = fit_transform_pca(X_train, X_train, X_test, n_components=best["pca"])
    # Train kNN
    knn = KNNFromScratch(k=best["k"], metric=best["metric"])
    knn.fit(Xtr_p, y_train)
    pred = knn.predict(Xte_p)
    test_acc = accuracy_score(y_test, pred)
    return test_acc

raw_test_acc = final_train_and_test(Xtr_raw, ytr, Xte_raw, yte, best_raw)
hog_test_acc = final_train_and_test(Xtr_hog, ytr, Xte_hog, yte, best_hog)

print(f"\n[Test Accuracy] RAW best model : {raw_test_acc:.4f}")
print(f"[Test Accuracy] HOG best model : {hog_test_acc:.4f}")

# ----------------------------
# Print compact results table
# ----------------------------
def summarize(history):
    # sort by cv_acc desc
    s = sorted(history, key=lambda r: -r["cv_acc"])[:12]
    print("\nTop configs:")
    print("feature  pca  metric      k   cv_acc")
    for r in s:
        print(f"{r['feature']:6}  {r['pca']:3}  {r['metric']:9}  {r['k']:2}  {r['cv_acc']:.4f}")

summarize(hist_raw)
summarize(hist_hog)

print("\n[Done] Plots saved:")
print(" - ./figs/val_acc_vs_k_RAW.png")
print(" - ./figs/val_acc_vs_k_HOG.png")
