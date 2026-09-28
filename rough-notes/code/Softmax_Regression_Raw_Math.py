"""
Softmax regression, the mathematical layer: plain Python, no numpy/torch for
the actual math. Fashion-MNIST is downloaded once via torchvision (used only
to fetch and cache the raw files); everything below reads those bytes and
does forward pass, softmax, cross-entropy and gradients by hand.

See permanent-notes/Index.md#levels-of-abstraction - this is level 1, one
step below permanent-notes/02-linear-neural-network-for-classification/L1_From_Scratch.py.
"""

import math
import random
import struct
from pathlib import Path

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
        raw = f.read(n * rows * cols)
    size = rows * cols
    return [[b / 255.0 for b in raw[i * size:(i + 1) * size]] for i in range(n)]


def read_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return list(f.read(n))


class DataModule:
    """A small, shuffled subset of Fashion-MNIST; batches as plain Python lists."""

    def __init__(self, n_train=200, n_test=60, batch_size=20, seed=42):
        _ensure_downloaded()
        images = read_idx_images(DATA_DIR / "train-images-idx3-ubyte")
        labels = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte")

        rng = random.Random(seed)
        order = list(range(len(images)))
        rng.shuffle(order)

        self.batch_size = batch_size
        self.rng = rng
        self.train = [(images[i], labels[i]) for i in order[:n_train]]
        self.test = [(images[i], labels[i]) for i in order[n_train:n_train + n_test]]

    def train_batches(self):
        """Yield shuffled mini-batches for one epoch."""
        shuffled = self.train[:]
        self.rng.shuffle(shuffled)
        for start in range(0, len(shuffled), self.batch_size):
            yield shuffled[start:start + self.batch_size]


class Model:
    """10 linear units sharing one softmax; W[j] holds class j's 784 weights."""

    NUM_INPUTS = 28 * 28
    NUM_CLASSES = 10

    def __init__(self, sigma=0.01, seed=1):
        rng = random.Random(seed)
        self.W = [[rng.gauss(0, sigma) for _ in range(self.NUM_INPUTS)]
                  for _ in range(self.NUM_CLASSES)]
        self.b = [0.0] * self.NUM_CLASSES

    def logits(self, x):
        return [sum(w_j[i] * x[i] for i in range(self.NUM_INPUTS)) + self.b[j]
                for j, w_j in enumerate(self.W)]

    def softmax(self, z):
        # Subtract the max before exp: same result (softmax is shift-invariant),
        # but math.exp() raises OverflowError on large inputs in plain Python
        # (numpy/torch would just return inf). This is the same idea behind
        # LogSumExp - see rough-notes/Rough_Note_03.md and L2_PyTorch.py, where
        # nn.CrossEntropyLoss applies it and L1_From_Scratch.py deliberately
        # does not (matching the book's naive "scratch" version).
        m = max(z)
        exps = [math.exp(v - m) for v in z]
        total = sum(exps)
        return [e / total for e in exps]

    def predict_proba(self, x):
        return self.softmax(self.logits(x))

    def train_batch(self, batch, lr):
        grad_W = [[0.0] * self.NUM_INPUTS for _ in range(self.NUM_CLASSES)]
        grad_b = [0.0] * self.NUM_CLASSES
        total_loss = 0.0

        for x, y in batch:
            probs = self.predict_proba(x)
            total_loss += -math.log(max(probs[y], 1e-12))

            # d(cross_entropy o softmax)/d(logit_j) collapses to (y_hat_j - y_j),
            # y_j being the one-hot label. Derivation in Rough_Note_03.md.
            for j in range(self.NUM_CLASSES):
                error = probs[j] - (1.0 if j == y else 0.0)
                grad_b[j] += error
                row = grad_W[j]
                for i in range(self.NUM_INPUTS):
                    row[i] += error * x[i]

        n = len(batch)
        for j in range(self.NUM_CLASSES):
            self.b[j] -= lr * grad_b[j] / n
            row, g_row = self.W[j], grad_W[j]
            for i in range(self.NUM_INPUTS):
                row[i] -= lr * g_row[i] / n

        return total_loss / n

    def evaluate(self, dataset):
        """Average loss and accuracy over a list of (x, y) pairs."""
        loss, correct = 0.0, 0
        for x, y in dataset:
            probs = self.predict_proba(x)
            loss += -math.log(max(probs[y], 1e-12))
            predicted = max(range(self.NUM_CLASSES), key=lambda j: probs[j])
            correct += int(predicted == y)
        return loss / len(dataset), correct / len(dataset)


class Trainer:
    """Runs mini-batch SGD and records train/test loss + accuracy per epoch."""

    def __init__(self, model, data, lr=0.1, epochs=30):
        self.model = model
        self.data = data
        self.lr = lr
        self.epochs = epochs

    def fit(self, verbose=True):
        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(1, self.epochs + 1):
            for batch in self.data.train_batches():
                self.model.train_batch(batch, self.lr)

            train_loss, train_acc = self.model.evaluate(self.data.train)
            test_loss, test_acc = self.model.evaluate(self.data.test)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if verbose:
                print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.2%} "
                      f"| test loss {test_loss:.4f} acc {test_acc:.2%}")

        return history


def main():
    data = DataModule()
    model = Model()
    Trainer(model, data).fit()

    x, y = data.test[0]
    probs = model.predict_proba(x)
    pred = max(range(Model.NUM_CLASSES), key=lambda j: probs[j])
    print(f"\nsample prediction: true={CLASS_NAMES[y]}  predicted={CLASS_NAMES[pred]}")


if __name__ == "__main__":
    main()
