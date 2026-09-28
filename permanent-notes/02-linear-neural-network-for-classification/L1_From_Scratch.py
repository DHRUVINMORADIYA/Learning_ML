"""
softmax, cross-entropy, forward pass, hand-derived gradients, manual SGD update
"""

import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "FashionMNIST" / "raw"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _ensure_downloaded():
    if DATA_DIR.exists() and any(DATA_DIR.glob("*-images-idx3-ubyte")):
        return
    import torchvision  # only used here, to fetch/cache the files once
    torchvision.datasets.FashionMNIST(root=str(DATA_DIR.parents[1]), download=True)


def read_idx_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path):
    with open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


class FashionMNISTData:
    """Fashion-MNIST read straight from the idx files as numpy arrays."""

    def __init__(self, batch_size=256, seed=1):
        _ensure_downloaded()
        self.X_train = read_idx_images(DATA_DIR / "train-images-idx3-ubyte")
        self.y_train = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte")
        self.X_test = read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte")
        self.y_test = read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte")
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def train_batches(self):
        """Yield shuffled mini-batches for one epoch."""
        order = self._rng.permutation(len(self.X_train))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            yield self.X_train[idx], self.y_train[idx]


def softmax(Z):
    """Z: (batch, 10) logits -> (batch, 10) probabilities.

    Naive on purpose (matches the book's scratch version): exp() can overflow
    for large logits. L2_PyTorch.py's nn.CrossEntropyLoss avoids this via the
    LogSumExp trick - see rough-notes/Rough_Note_03.md.
    """
    Z_exp = np.exp(Z)
    return Z_exp / Z_exp.sum(axis=1, keepdims=True)


def cross_entropy(y_hat, y):
    """y_hat: (batch, 10) probabilities, y: (batch,) class indices."""
    return -np.mean(np.log(y_hat[np.arange(len(y_hat)), y]))


class SoftmaxRegression:
    """784 inputs -> 10 classes. W, b are plain arrays; no autograd involved."""

    NUM_INPUTS = 28 * 28
    NUM_CLASSES = 10

    def __init__(self, sigma=0.01, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, sigma, size=(self.NUM_INPUTS, self.NUM_CLASSES))
        self.b = np.zeros(self.NUM_CLASSES)

    def forward(self, X):
        return softmax(X @ self.W + self.b)

    def loss(self, X, y):
        return cross_entropy(self.forward(X), y)

    def accuracy(self, X, y):
        return np.mean(self.forward(X).argmax(axis=1) == y)

    def gradients(self, X, y):
        """d(cross_entropy o softmax)/d(logit) collapses to (y_hat - y_onehot);
        derivation in rough-notes/Rough_Note_03.md. Taken as given here, not
        re-derived by a framework."""
        n = len(y)
        y_hat = self.forward(X)
        Y_onehot = np.zeros_like(y_hat)
        Y_onehot[np.arange(n), y] = 1.0
        dZ = (y_hat - Y_onehot) / n

        grad_W = X.T @ dZ
        grad_b = dZ.sum(axis=0)
        return grad_W, grad_b


class Trainer:
    """Runs mini-batch SGD and records train/test loss + accuracy per epoch."""

    def __init__(self, model, data, lr=0.1, epochs=10):
        self.model = model
        self.data = data
        self.lr = lr
        self.epochs = epochs

    def fit(self, verbose=True):
        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(1, self.epochs + 1):
            for X, y in self.data.train_batches():
                grad_W, grad_b = self.model.gradients(X, y)
                self.model.W -= self.lr * grad_W
                self.model.b -= self.lr * grad_b

            train_loss = self.model.loss(self.data.X_train, self.data.y_train)
            test_loss = self.model.loss(self.data.X_test, self.data.y_test)
            train_acc = self.model.accuracy(self.data.X_train, self.data.y_train)
            test_acc = self.model.accuracy(self.data.X_test, self.data.y_test)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if verbose:
                print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.2%} "
                      f"| test loss {test_loss:.4f} acc {test_acc:.2%}")

        return history


def plot_history(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history["train_loss"], label="train")
    ax_loss.plot(history["test_loss"], "--", label="test")
    ax_loss.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()

    ax_acc.plot(history["train_acc"], label="train")
    ax_acc.plot(history["test_acc"], "--", label="test")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()


def show_predictions(model, data, n=8):
    X, y = data.X_test[:n], data.y_test[:n]
    preds = model.forward(X).argmax(axis=1)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[preds[i]]}\n(true: {CLASS_NAMES[y[i]]})", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    data = FashionMNISTData()
    model = SoftmaxRegression()
    history = Trainer(model, data).fit()

    plot_history(history)
    show_predictions(model, data)


if __name__ == "__main__":
    main()
