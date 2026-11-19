---
title: Rough Note 01
parent: Rough Notes
nav_order: 1
---

### Why do we need Artificial Neural Networks?
    
In traditional setup, programs are developed based on rules. Devs receive a set of rules, they convert them into code and there - we have a software. Imagine developers paving way for input data to take it to the expected outcome.

Concepts of ML and ANN come into picture when the requirement doesn't have defined rules. For example, predicting weather or identifying potential fraud in financial data. We rather train models with huge amount of data. Imagine data(like water) creating its own path to get to the best possible outcome.

Model is nothing but a set of flexible inter-connected knobs that we can tune in order to deliver the best possible result in any case.

This code is my attempt at understanding a typical training cycle (epoch) in a neural network with below mentioned concepts without going into metrices yet.  [`code/ANN_single_training_step_simulation.py`](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/rough-notes/code/ANN_single_training_step_simulation.py){:target="_blank"}.
- Front propagation (weighted sums + biases, sigmoid)
- Loss calculation (squared error) and 
- Back propagation (partial derivation, chain rule, gradient descent)  

### Vector-Matrix multiplication & Matrix-Matrix multiplication

The previous section gave idea of a simple instance of a typical training process. There were two inputs with single hidden layer with two neurons.

Off course we need more layers and more neurons to make it capable of finding patterns in unorganized data.

To represent and calculate billions of values, metrices come handy for its convenience and efficiency that they unlock when used with GPUs due to parallel processing abilities.

Matrix multiplication steps are not that hard, however understanding them from core is a bit tricky. Here is my idea of it:

B = A @ X where B and A are metrices and X is an input vector; @ is multiplication.

In this example, A works as an actor and X is a subject. A will transform X into B.

We can imagine this from other pov as well where A and X are both transformers and we are proactively calculating its combined effect and storing it in B to use it in future on some input.

Let's say A and X both are our actors waiting for some input Y in future.
So we can say: (A @ X) @ Y = A @ (X @ Y).
Sequence does matter.

How does this apply in neural networks?
Imagine a network with 2 inputs and a hidden layer with 5 neurons in it.
there will be 10 weights in total which can be presented with 5x2 matrix W.

let's say input matrix 2x1 is called X, so weighted sum can be simply calculated with W @ X. Also note how 2 input values were moved forward and converted into 5 values.

If input vector was a matrix rather than a vector, we can say we are doing the same process for multiple input sets and process follows the same way.

### Full-batch processing & mini-batch processing & stochastic gradient descent

Given that training data count is in thousands or millions, there comes different ways we can feed them to models. Off course not all ways are practical and advisable due to both accuracy and computational reasons.

First way is to feed all training data together, get average of squared error loss and pass the loss using below 2 options:
1. If it is linear regression model, there is a formula (Analytic Solution) derived out of differantion of squared error loss function. Very quick and easy - it will give the best possible updated values for weights and biases. However it only works with linear models where there are no activation functions.
2. Now let's say ours is non-linear. In that case we use gradient descent. Same way as loss, we can get average gradient value of all data points and move accordingly. This one doesn't seem practical off course.

Let's think opposite and run this cycle separately for each data. 
1. Here if given a chance and we try using analytic solution, we would wipe out word done in previous cycle and overwrite it with best possible tuning for current cycle. Not good.
2. We can however do gradient descent. We are saying let's move slightly each time in a direction which would take us close to bottom of n-dimensional graph of loss vs all parameters. Here drawback is (1) we become too much sensitive and be bothered by "noise" data (imagine data falling on edges of gaussian distribution). (2) takes high CPU.

Practical and I suppose the industry practice should be somewhere in between above two - do it in batches.
1. Doesn't make sense to use analytic solution (if we are lucky to have linear regression), as each batch run will basically wipe out work done by previous one.
2. We use gradient descent here. We take average of loss as well as of gradient, use learning rate to define step sizes we take and move slightly with each run. This one feels the best.

So, mini-batch runs with gradient descent method it is!
