---
title: Rough Note 04
parent: Rough Notes
nav_order: 4
---

Chapter 5: Multilayer Perceptrons

This is basically from where I started this - ANN_single_epoch_simulation.py and MNIST image classfication problems.

In last two chapters there was a single step to derive output. All input points affects directly the output.
MLP introduces hidden layer where all neurons from previous layer directly affects all neurons in next layer.

Why layers?
If we keep single layer with huge amount of neurons, they will all be directly affected by each input. Theoretically it can work, but training process will become more tedious. In multiple layers, we can imagine all neurons looking for their "thing" in previous layer. Pattern recognition becomes more intelligent when each neurons in first layer starts to identify certain features and quirks (example, curves, lines, etc). The next layer now has access to high level conclusion derived by first layer and they make further decisions.

Activation functions
If no activation functions are used, many layers at the end become a single function. By adding activation function we add non-linearity. Imagine a flat line/plan trying to find pattern in 2d/3d graphs vs a flexible line/plan tring to find pattern in 2d/3d space. Activation functions will give this flexibility to move around to fit data.

ReLU - rectified linear unit

ReLU(x) = max(x, 0)

It is straightforward. We allow the value as it is if it is positive and make it zero if it is negative.

another option? parametrized ReLU

pReLU = max(0, x) + a x min(0, x)

We let negatives come in but in proportion of a fixed constant a.

Benefits and drawbacks of ReLU
gradients become either 0 or 1. means gradient passes through without being shrunk. Either entire gradient or nothing.
It helps mitigating vanishing gradient problem.
On the other hand, when it is zero, that neuron doesn't learn anything and it doesn't pass anything back to previous layer. It is called dead ReLU problem.

Sigmoid

1 / 1 + exp(-x)

Sigmoid converts any value in a value between 0 to 1. 
as values get bigger sigmoid lean towards 1 and as they get smaller sigmoid goes towards 0. with input value 0, sigmoid becomes 0.5.
Sigmoid seems to be new thing after ReLU. Rather than just saying if a neuron fired or no, we give more info about in what intensity it fired. kinda

drawback is vanishing gradients. as we pass through multiple layers, gradient keeps getting smaller and smaller.

derivative of sigmoid is = sigmoid x (1- sigmoid)

Tanh

tanh = 1 - exp(-2x) / 1 + exp(-2x)

similar to sigmoid but given output distribution lies between -1 and 1.

nothing much mentioned about good, bad and usecases of tanh.

There are more activation functions out there GELU and more. more later.