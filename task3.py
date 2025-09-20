"""
Task 3: SVM with Linear Model, Kernel Trick, and Hyperparameter Search
- Linear SVM from scratch (hinge loss, L2 reg), One-vs-Rest for 10 classes
- Kernel SVM (RBF) via sklearn SVC + GridSearchCV over C & gamma
- Validation accuracy heatmap, runtime & test accuracy comparison
"""

import os, time, random, numpy as np, matplotlib
matplotlib.use("Agg")  # save plots
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ---------------------------
# Config (tweak for speed)
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = "./cifar10data"
FIGS = "./figs"; os.makedirs(FIGS, exist_ok=True)

USE_HOG = True       # True -> HOG features, False -> RAW pixels
N_TRAIN = 3000       # subset sizes to keep it reasonable on CPU
N_VAL   = 1000
N_TEST  = 2000
PCA_DIM = 100        # feature dim for both methods (after StandardScaler)

# HOG params
HOG_KW = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
              visualize=False, block_norm="L2-Hys")

# Grid for Kernel SVM
C_GRID = [0.01, 0.1, 1, 10, 100]
GAMMA_GRID = [0.001, 0.01, 0.1, 1]

# ---------------------------
# Load CIFAR-10 & build features
# ---------------------------
print("[Info] Loading CIFAR-10 & sampling subsets...")
transform = transforms.ToTensor()
train_full = datasets.CIFAR10(root=ROOT, train=True,  download=True, transform=transform)
test_set   = datasets.CIFAR10(root=ROOT, train=False, download=True, transform=transform)
classes = train_full.classes

train_set, val_set = random_split(train_full, [45000, 5000],
                                  generator=torch.Generator().manual_seed(SEED))

def take_subset(ds, n):
    idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)
    X, y = [], []
    for i in idxs:
        img_t, lab = ds[i]
        X.append(img_t.numpy())  # C,H,W in [0,1]
        y.append(lab)
    return np.array(X), np.array(y)

Xtr_img, ytr = take_subset(train_set, N_TRAIN)
Xval_img, yval = take_subset(val_set, N_VAL)
Xte_img, yte = take_subset(test_set, N_TEST)

def to_raw(X_img):
    N = X_img.shape[0]
    return X_img.reshape(N, -1).astype(np.float32)

def to_hog(X_img):
    feats = []
    for img in X_img:
        img_hwc = np.transpose(img, (1,2,0))
        gray = rgb2gray(img_hwc)
        feats.append(hog(gray, **HOG_KW).astype(np.float32))
    return np.vstack(feats)

if USE_HOG:
    print("[Info] Using HOG features…")
    Xtr, Xval, Xte = to_hog(Xtr_img), to_hog(Xval_img), to_hog(Xte_img)
else:
    print("[Info] Using RAW flattened pixels…")
    Xtr, Xval, Xte = to_raw(Xtr_img), to_raw(Xval_img), to_raw(Xte_img)

# Standardize + PCA for both Linear & Kernel SVM
scaler = StandardScaler()
pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SEED)

Xtr_s = scaler.fit_transform(Xtr)
Xval_s = scaler.transform(Xval)
Xte_s  = scaler.transform(Xte)

Xtr_p = pca.fit_transform(Xtr_s)
Xval_p = pca.transform(Xval_s)
Xte_p  = pca.transform(Xte_s)

print(f"[Info] Feature dims -> raw:{Xtr.shape[1]}  after PCA:{Xtr_p.shape[1]}")

# --------------------------------------
# Linear SVM from scratch (One-vs-Rest)
# --------------------------------------
class LinearSVM_OVR:
    """
    Multiclass Linear SVM via One-vs-Rest with hinge loss:
    Min_w (1/2)||w||^2 + C * mean(max(0, 1 - y * (Xw + b)))
    Trained with mini-batch gradient descent (L2 reg).
    """
    def __init__(self, n_classes, lr=1e-2, C=1.0, epochs=15, batch_size=128, verbose=True):
        self.n_classes = n_classes
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.W = None   # (K, d)
        self.b = None   # (K,)

    def _train_binary(self, X, y):
        # y in {+1, -1}
        n, d = X.shape
        w = np.zeros(d, dtype=np.float32)
        b = 0.0
        lr = self.lr
        C = self.C
        bs = self.batch_size

        for ep in range(self.epochs):
            perm = np.random.permutation(n)
            Xsh, ysh = X[perm], y[perm]
            for i in range(0, n, bs):
                xb = Xsh[i:i+bs]
                yb = ysh[i:i+bs]  # (+1/-1)
                # margins: y * (x·w + b)
                margin = yb * (xb @ w + b)
                mask = (margin < 1).astype(np.float32)
                # gradients
                # d/dw: w - C * mean( y_i * x_i [margin<1] )
                if len(xb) > 0:
                    grad_w = w - C * ( (yb[:,None] * xb * mask[:,None]).mean(axis=0) )
                    grad_b = - C * ( (yb * mask).mean() )
                else:
                    grad_w = w
                    grad_b = 0.0
                # update
                w -= lr * grad_w
                b -= lr * grad_b
        return w, b

    def fit(self, X, y):
        K = self.n_classes
        n, d = X.shape
        self.W = np.zeros((K, d), dtype=np.float32)
        self.b = np.zeros(K, dtype=np.float32)
        for k in range(K):
            y_bin = np.where(y == k, 1.0, -1.0)  # class k vs rest
            if self.verbose:
                print(f"[Linear-SVM] Training OvR classifier for class {k}/{K-1}…")
            wk, bk = self._train_binary(X, y_bin)
            self.W[k] = wk
            self.b[k] = bk

    def decision_function(self, X):
        # scores per class
        return X @ self.W.T + self.b[None, :]

    def predict(self, X):
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

# Train Linear SVM (scratch)
t0 = time.time()
lin_svm = LinearSVM_OVR(n_classes=len(classes), lr=1e-2, C=1.0, epochs=15, batch_size=128, verbose=True)
lin_svm.fit(Xtr_p, ytr)
lin_train_time = time.time() - t0

# Eval Linear SVM
t1 = time.time()
yval_pred_lin = lin_svm.predict(Xval_p)
yte_pred_lin  = lin_svm.predict(Xte_p)
lin_pred_time = time.time() - t1
lin_val_acc = accuracy_score(yval, yval_pred_lin)
lin_test_acc = accuracy_score(yte, yte_pred_lin)
print(f"[Linear SVM] val_acc={lin_val_acc:.4f}  test_acc={lin_test_acc:.4f}  train_time={lin_train_time:.1f}s  pred_time={lin_pred_time:.1f}s")

# --------------------------------------
# Kernel SVM (RBF) via sklearn + GridSearchCV
# --------------------------------------
print("[Kernel SVM] Grid search over C and gamma …")
param_grid = {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID}
pipe = Pipeline([
    ("scaler", StandardScaler()),   # scale again on PCA-space? (safe, cheap)
    ("svc", SVC(kernel="rbf", decision_function_shape="ovr"))
])

# We grid-search on the PCA-transformed features (Xtr_p)
t2 = time.time()
grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=0)
grid.fit(Xtr_p, ytr)
kernel_search_time = time.time() - t2

best_C = grid.best_params_["svc__C"]
best_gamma = grid.best_params_["svc__gamma"]
best_cv_acc = grid.best_score_
print(f"[Kernel SVM] best C={best_C}, best gamma={best_gamma}, cv_acc={best_cv_acc:.4f}")

# Evaluate on val/test with best model
t3 = time.time()
yval_pred_k = grid.predict(Xval_p)
yte_pred_k  = grid.predict(Xte_p)
kernel_pred_time = time.time() - t3
kernel_val_acc = accuracy_score(yval, yval_pred_k)
kernel_test_acc = accuracy_score(yte, yte_pred_k)
print(f"[Kernel SVM] val_acc={kernel_val_acc:.4f}  test_acc={kernel_test_acc:.4f}  search_time={kernel_search_time:.1f}s  pred_time={kernel_pred_time:.1f}s")

# --------------------------------------
# Heatmap of validation accuracy for (C, gamma)
# --------------------------------------
# Build a grid table of mean CV accuracy
means = grid.cv_results_["mean_test_score"]
params = grid.cv_results_["params"]
# map to (gamma, C) matrix
gamma_list = GAMMA_GRID
C_list = C_GRID
heat = np.zeros((len(gamma_list), len(C_list)), dtype=np.float32)
for m, p in zip(means, params):
    c = p["svc__C"]; g = p["svc__gamma"]
    i = gamma_list.index(g); j = C_list.index(c)
    heat[i, j] = m

plt.figure(figsize=(7,5))
im = plt.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(im, fraction=0.046, pad=0.04, label="CV Accuracy")
plt.xticks(range(len(C_list)), C_list)
plt.yticks(range(len(gamma_list)), gamma_list)
plt.xlabel("C"); plt.ylabel("gamma")
plt.title(f"Validation Accuracy (RBF SVM, PCA={PCA_DIM}, {'HOG' if USE_HOG else 'RAW'})")
for i in range(len(gamma_list)):
    for j in range(len(C_list)):
        plt.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
plt.tight_layout()
out_path = os.path.join(FIGS, f"svm_rbf_heatmap_pca{PCA_DIM}_{'HOG' if USE_HOG else 'RAW'}.png")
plt.savefig(out_path, dpi=150); plt.close()
print(f"[Saved] Heatmap -> {out_path}")

# --------------------------------------
# Final comparison summary
# --------------------------------------
print("\n=== Runtime & Accuracy Comparison ===")
print(f"Linear SVM (scratch): val={lin_val_acc:.4f}, test={lin_test_acc:.4f}, train_time={lin_train_time:.1f}s")
print(f"Kernel SVM (RBF):     val={kernel_val_acc:.4f}, test={kernel_test_acc:.4f}, search_time={kernel_search_time:.1f}s")


