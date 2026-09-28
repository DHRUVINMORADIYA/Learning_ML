---
title: Permanent Notes
nav_order: 2
has_children: true
---

# Permanent Notes · Index

## Path

1. **[00 · Basics](00-basics/Notes.md)**: from "why learn from data" to how a
   network learns. The four components, neuron to network, loss, gradient descent,
   matrices, autograd, batching.
2. **[01 · Linear Neural Network for Regression](01-linear-neural-network-for-regression/Notes.md)**:
   the first model end to end, built at three levels of abstraction; generalization
   and weight decay.
3. **[02 · Linear Neural Network for Classification](02-linear-neural-network-for-classification/Notes.md)**:
   the same linear model predicting a class instead of a number. Softmax,
   cross-entropy, LogSumExp; test set sizing, VC dimension, distribution shift.

## Concepts

Core ideas kept at a broad level here and cited from the topic notes.

### Levels of abstraction

When working with a model there are four layers of abstraction, from most manual to
most hands-off. Each step hands one more responsibility to a library.

1. **Mathematical layer**: linear algebra, activation functions, loss functions,
   calculus. Gradients are derived by hand (chain rule).
2. **From scratch**: the derived formulas are taken as given and applied (e.g.
   plugging into the gradient expression). Still no framework.
3. **Framework, control kept**: a library (PyTorch, TensorFlow) computes gradients
   via autograd. How much to hand over is a choice: raw autograd (own parameter
   creation and updates) or `nn` + `optim` (library does those) while still writing
   the training loop.
4. **High level**: a framework (Keras, Lightning, fastai) runs everything,
   including the loop. Only data and parameters are configured.

Cited by: [01 · Linear Neural Network for Regression](01-linear-neural-network-for-regression/Notes.md),
[02 · Linear Neural Network for Classification](02-linear-neural-network-for-classification/Notes.md)

### Four components of a learning system

Any supervised-learning setup decomposes into four parts, and most questions are
really about just one of them:

1. **Data**: training, validation, test sets.
2. **Model**: the function mapping input to prediction.
3. **Objective (loss) function**: a single number for how wrong a prediction is.
4. **Optimization algorithm**: the rule that updates the model's parameters.

Cited by: [00 · Basics](00-basics/Notes.md)

### Feeding data: full-batch / stochastic / mini-batch

How many examples go into one parameter update, a precision vs cost trade-off.

- **Full-batch**: all data per update; accurate but expensive. Linear models with
  no activation also admit a closed-form analytic solution.
- **Stochastic**: one example per update; cheap, noisy, outlier-sensitive.
- **Mini-batch**: a handful per update; averages out noise, fits in memory. The
  default.

Cited by: [00 · Basics](00-basics/Notes.md)
