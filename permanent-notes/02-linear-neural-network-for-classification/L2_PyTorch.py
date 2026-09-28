"""
The PyTorch layer.

Same softmax regression as L1_From_Scratch.py. nn.Linear now holds W/b,
nn.CrossEntropyLoss computes softmax + cross-entropy together (numerically
safe via the LogSumExp trick - see rough-notes/Rough_Note_03.md, where the
naive softmax-then-log version was worked out by hand), and optim.SGD applies
the update. We still write the training loop.

--- one level deeper (what nn.Linear / nn.CrossEntropyLoss / optim.SGD wrap) --
    W = torch.normal(0, 0.01, (784, 10), requires_grad=True)   # instead of nn.Linear
    b = torch.zeros(10, requires_grad=True)
    logits = X @ W + b                                          # instead of model(X)

    m = logits.max(dim=1, keepdim=True).values                 # LogSumExp shift
    log_probs = logits - m - (logits - m).exp().sum(1, keepdim=True).log()
    loss = -log_probs[range(len(y)), y].mean()                 # instead of CrossEntropyLoss

    loss.backward()                                             # this part is the same
    with torch.no_grad():                                       # instead of optimizer.step()
        for p in (W, b):
            p -= lr * p.grad
            p.grad.zero_()                                      # instead of optimizer.zero_grad()
---------------------------------------------------------------------------
"""

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNISTData:
    """Fashion-MNIST, each 28x28 image flattened to a 784-length vector."""

    def __init__(self, batch_size=256, root="./data"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.reshape(-1)),
        ])
        raw_train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        raw_test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)

        self.batch_size = batch_size
        self.X_train, self.y_train = self._materialize(raw_train)
        self.X_test, self.y_test = self._materialize(raw_test)
        self.train_ds = TensorDataset(self.X_train, self.y_train)

    @staticmethod
    def _materialize(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        return next(iter(loader))

    def train_loader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def full(self, train=True):
        return (self.X_train, self.y_train) if train else (self.X_test, self.y_test)


class SoftmaxRegression(nn.Module):
    """784 inputs -> 10 class logits (no softmax here - the loss fuses it in)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(28 * 28, 10)

    def forward(self, X):
        return self.net(X)


class Trainer:
    """Mini-batch SGD via autograd; records train/test loss + accuracy per epoch."""

    def __init__(self, model, data, lr=0.1, epochs=10):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    @torch.no_grad()
    def _evaluate(self, X, y):
        logits = self.model(X)
        loss = self.loss_fn(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss, acc

    def fit(self, verbose=True):
        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(1, self.epochs + 1):
            for X, y in self.data.train_loader():
                self.optimizer.zero_grad()               # deeper: p.grad.zero_()
                loss = self.loss_fn(self.model(X), y)
                loss.backward()                           # same at every level
                self.optimizer.step()                     # deeper: p -= lr * p.grad

            train_loss, train_acc = self._evaluate(*self.data.full(train=True))
            test_loss, test_acc = self._evaluate(*self.data.full(train=False))
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
    X, y = data.full(train=False)
    X, y = X[:n], y[:n]
    preds = model(X).argmax(dim=1)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[preds[i]]}\n(true: {CLASS_NAMES[y[i]]})", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(0)
    data = FashionMNISTData()
    model = SoftmaxRegression()
    history = Trainer(model, data).fit()

    plot_history(history)
    show_predictions(model, data)


if __name__ == "__main__":
    main()
