"""
The high-level layer: the framework runs the loop too.

Same softmax regression again, but there is no training loop in this file.
We describe the model, compile it with an optimizer + loss, and call
model.fit(). Keras owns the epoch/batch/backward/step cycle.

from_logits=True keeps softmax fused into the loss (LogSumExp-safe), matching
L2_PyTorch.py's nn.CrossEntropyLoss - see rough-notes/Rough_Note_03.md for the
derivation of why that fusion matters.

(PyTorch-ecosystem equivalents: PyTorch Lightning, skorch, fastai.)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    """Fashion-MNIST, each 28x28 image flattened to a 784-length vector."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
    return X_train, y_train, X_test, y_test


def build_model(lr=0.1):
    """784 inputs -> 10 class logits (no softmax layer - the loss fuses it in)."""
    model = keras.Sequential([
        keras.layers.Input((28 * 28,)),
        keras.layers.Dense(10),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], "--", label="test")
    ax_loss.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()

    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], "--", label="test")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()


def show_predictions(model, X_test, y_test, n=8):
    logits = model.predict(X_test[:n], verbose=0)
    preds = logits.argmax(axis=1)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[preds[i]]}\n(true: {CLASS_NAMES[y_test[i]]})", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    keras.utils.set_random_seed(0)
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=256, epochs=10, verbose=0,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"final test loss: {test_loss:.4f}  test accuracy: {test_acc:.2%}")

    plot_history(hist.history)
    show_predictions(model, X_test, y_test)


if __name__ == "__main__":
    main()
