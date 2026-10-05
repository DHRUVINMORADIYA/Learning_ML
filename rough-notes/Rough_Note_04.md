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

5.2 Implementation of Multilayer Perceptrons
We use python sequential to just say how many layers, width of each layer and pytorch gives us that without worrying about inner architecture.

back prop revision as it came again in book

Derived delta J / delta W2 and same for W1 on paper. here is what I am understanding. correct if it is different. delta L / delta o is important one that will be carried over all the way down. it is basically derivative of loss function with reference to prediction made. for sme it is 2/n (pred i - y). if we happen to use softmax and cross entropy, it is simply ( y hat - y). for the output layer updates, we just multiply it with their corresponding h values. he values are values after activation functions applied on previous layer neurons. as we have multiple weights, they all will receive their own h values. now coming to hidden layer. delta L / delta o(d value) will come here as it is. we first multiply it with w2 value. it will basically shrink the d value a bit. w2 is like a gatekeeper - not letting entire thing go to the end. so same way the same strictness should be applied when it comes to who gets more punishment. we then multiply this with x values. we do transpose. not really clear why. but it then matches shape when we have metrices. here also we are making punishment in proportion to who contributed what. we then multiple it with derivative of activation function. if it is relu, it becomes simple 0 or 1. if it is sigmoid. it becomes h x (1-h).

5.4 Numerical Stability and Initialization
- This section is about maintaining values of weights and biases and keeping them sane.

1. Vanishing Gradients issue
- Use of sigmoid comes up with this risk. Although it is considered as a similar thresholding activation function to human brain, it can make gradients zero. Reason? it's values are squished between 1 and 0. now when values go through multiple layers, values keep shrinking and it may cause this issue. ReLU on other hand is better at this as we are allowing entire value to pass through. Only concern with ReLU is dying ReLU situation where if some nerons become zero, it becomes useless. There are although some mitigation methods for that.

2. Exploding gradients issue
- similar mirror case where values keep growing and causes memory overflow issues.

3. Breaking the symmetry
- It is important to assign random values to weights and biases at beginning - to make model expressive.