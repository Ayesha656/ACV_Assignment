import random, numpy as np, torch, torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---- Repro & device
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- CIFAR-10 stats (for standardization)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---- Transforms
# 1) Base to-tensor (scales to [0,1]) + normalize
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# 2) Augmentation: random crop w/ padding, hflip, color jitter
augment_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ---- Download/load CIFAR-10
root = "./cifar10data"
train_full = datasets.CIFAR10(root=root, train=True, download=True, transform=base_transform)
test_set   = datasets.CIFAR10(root=root, train=False, download=True, transform=base_transform)
classes = train_full.classes  # list of class names

# ---- Split train into train/val (e.g., 45k train / 5k val)
val_size = 5000
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

# ---- Dataloaders (feel free to tune batch_size)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# ---- For augmentation visualization, make an "augmented view" of the training data
train_aug_view = datasets.CIFAR10(root=root, train=True, download=False, transform=augment_transform)

# =========================
# Helper: show a grid of images with labels
# =========================
def imshow_grid(tensor_images, labels, classes, ncols=10, title=None):
    n = len(tensor_images)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(ncols*1.6, nrows*1.8))
    for i, (img_t, lab) in enumerate(zip(tensor_images, labels)):
        plt.subplot(nrows, ncols, i+1)
        # denormalize for display: x*std + mean
        img = img_t.clone()
        for c in range(3):
            img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
        img = img.clamp(0,1).permute(1,2,0).numpy()
        plt.imshow(img)
        plt.title(classes[lab], fontsize=8)
        plt.axis('off')
    if title:
        plt.suptitle(title, y=0.98, fontsize=12)
    plt.tight_layout()
    plt.show()

# =========================
# A) Visualize 20 random original samples
# =========================
idxs = np.random.choice(len(train_set), size=20, replace=False)
original_imgs, original_labs = [], []
for i in idxs:
    img_t, lab = train_set[i]
    original_imgs.append(img_t.cpu())
    original_labs.append(lab)
imshow_grid(original_imgs, original_labs, classes, ncols=10, title="20 Random Original Samples")

# =========================
# B) Visualize 10 augmented samples (with the same labels)
# =========================
idxs_aug = np.random.choice(len(train_aug_view), size=10, replace=False)
aug_imgs, aug_labs = [], []
for i in idxs_aug:
    img_t, lab = train_aug_view[i]
    aug_imgs.append(img_t.cpu())
    aug_labs.append(lab)
imshow_grid(aug_imgs, aug_labs, classes, ncols=10, title="10 Augmented Samples")

# =========================
# C) HOG feature extraction (for kNN/SVM)
#    We'll create a function to turn a PIL image (or tensor) into HOG features.
#    For CIFAR-10 (32x32), HOG still works but is simple; we convert to gray.
# =========================
from torchvision.transforms.functional import to_pil_image

def tensor_to_display_np(img_t):
    """Denormalize + to HxWxC float [0,1] for HOG viz."""
    img = img_t.clone()
    for c in range(3):
        img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
    img = img.clamp(0,1).permute(1,2,0).numpy()
    return img

def hog_from_tensor(img_t, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
    # Convert normalized tensor -> [0,1] numpy -> grayscale
    img_np = tensor_to_display_np(img_t)
    gray = rgb2gray(img_np)
    feat, hog_img = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        block_norm='L2-Hys'
    )
    return feat, hog_img

# =========================
# D) Visualize HOG maps for the 10 augmented images
# =========================
plt.figure(figsize=(10, 10))
for i, img_t in enumerate(aug_imgs):
    _, hog_viz = hog_from_tensor(img_t)
    plt.subplot(5, 2, i+1)
    plt.imshow(hog_viz)
    plt.title(f"HOG of Augmented #{i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# =========================
# E) Build feature matrices (HOG) for a *small subset* for kNN/SVM demo
#    (You can scale up later; small subset keeps it fast.)
# =========================
def build_hog_features(dataset, n_samples=5000):
    idxs = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    X, y = [], []
    for i in idxs:
        img_t, lab = dataset[i]
        feat, _ = hog_from_tensor(img_t)
        X.append(feat)
        y.append(lab)
    return np.array(X), np.array(y)

print("Extracting HOG features (this may take a few minutes for 5k samples)...")
X_train, y_train = build_hog_features(train_set, n_samples=5000)
X_val,   y_val   = build_hog_features(val_set,   n_samples=2000)
X_test,  y_test  = build_hog_features(test_set,  n_samples=2000)

print("Train/Val/Test HOG shapes:", X_train.shape, X_val.shape, X_test.shape)

# =========================
# F) kNN & Linear SVM on HOG features
# =========================
# Pipeline: Standardize features -> classifier
knn_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)),
])

svm_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("svm", LinearSVC(C=1.0, max_iter=5000))
])

# Fit on train
print("Training kNN...")
knn_clf.fit(X_train, y_train)
print("Training SVM...")
svm_clf.fit(X_train, y_train)

# Evaluate
def evaluate(model, X, y, name=""):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    print(f"[{name}] Accuracy: {acc:.4f}")
    return acc, pred

print("\nValidation performance:")
_ = evaluate(knn_clf, X_val, y_val, name="kNN (val)")
_ = evaluate(svm_clf, X_val, y_val, name="SVM (val)")

print("\nTest performance:")
acc_knn, pred_knn = evaluate(knn_clf, X_test, y_test, name="kNN (test)")
acc_svm, pred_svm = evaluate(svm_clf, X_test, y_test, name="SVM (test)")

print("\nClassification Report (SVM on test):")
print(classification_report(y_test, pred_svm, target_names=classes, digits=4))
