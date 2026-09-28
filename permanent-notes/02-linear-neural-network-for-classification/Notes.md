---
title: 02 Linear Neural Network for Classification
parent: Permanent Notes
nav_order: 3
---

# 02 · Linear Neural Network for Classification

Builds on [01 · Linear Neural Network for Regression](../01-linear-neural-network-for-regression/Notes.md).

## Summary

Regression asks *how much*. Classification asks *which one*. The model itself barely changes; it is still the same linear function from chapter 01. Two things are new:

1. **Softmax** turns raw scores into a probability distribution.
2. **Cross-entropy** measures how wrong that distribution is.

Everything else stays as it was: forward pass, backpropagation, gradient descent, mini-batching.

Instead of one output we now produce one score per class,

$$
\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

where $\mathbf{o}$ holds one entry $o_j$ per class. These raw scores are called **logits**. They can be negative, they are unbounded, and they do not add up to anything meaningful. Softmax is what fixes that.

## Softmax

We need a probability distribution: every value positive, all of them adding to 1.

$$
\hat{y}_j = \frac{\exp(o_j)}{\sum_{k} \exp(o_k)}
$$

Two jobs are being done here:

- $\exp$ handles negatives, since it maps any number to a positive one. It also amplifies differences, so a small lead in logits becomes a bigger lead in probability.
- Dividing by the sum normalizes, so the outputs add to 1.

Softmax does not change which class wins; $\arg\max$ of the logits and $\arg\max$ of the probabilities give the same answer. What it buys us is an output we can push into a loss function that understands probability.

## Cross-entropy

The loss for a predicted distribution measured against the true one,

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j} y_j \log \hat{y}_j
$$

Labels are one-hot, so every term except the true class is multiplied by zero. The whole sum collapses to

$$
l = -\log \hat{y}_{\text{true class}}
$$

Predict 0.9 for the right class and the loss is small. Predict 0.01 and the loss is large. That is the entire behaviour.

Why $\log$? It spreads out the range between 0 and 1. It punishes hard as the probability approaches 0, and flattens to nothing as it approaches 1.

The formula is simple. Where it comes from is the interesting part, and there are two separate routes to it.

### Route 1: maximum likelihood estimation

We want to maximize the likelihood of the labels we actually saw, given the features. The examples are mutually independent, so we multiply,

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^{n} P(y^{(i)} \mid \mathbf{x}^{(i)})
$$

Multiplying many numbers below 1 drives the product toward zero, so we take a log and the product becomes a sum. Maximizing a likelihood is the same as minimizing its negative log,

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^{n} -\log P(y^{(i)} \mid \mathbf{x}^{(i)})
$$

and each term is exactly the cross-entropy of one example. This route explains the batch case naturally. Summing over examples was never a design choice; it fell out of the product.

### Route 2: entropy and information theory

**Surprisal** of an event is $-\log p$. If something certain happens we are not surprised at all ($-\log 1 = 0$). If something with a 1% chance happens we are very surprised.

**Entropy** is the average surprisal of a distribution, a quantification of how scattered it is,

$$
H(P) = -\sum_{j} p_j \log p_j
$$

Each surprisal is weighted by $p_j$ because a rare event is very surprising but rarely occurs; its share of the *average* has to be scaled by how often it actually shows up.

$[1, 0, 0]$ has zero entropy, total certainty. $[1/3, 1/3, 1/3]$ has the highest, no certainty at all.

**Cross-entropy** changes one thing: the weights come from the true distribution instead of the predicted one,

$$
H(P, Q) = -\sum_{j} p_j \log q_j
$$

We measure the *model's* surprisal, but weight it by *reality's* frequencies. With one-hot labels this is the same formula we started with.

Same destination, two roads. MLE explains the summing, entropy explains the $\log$.

## The gradient that matters

Substitute softmax into cross-entropy and the loss becomes

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = \log \sum_{k} \exp(o_k) - \sum_{j} y_j o_j
$$

Differentiate with respect to a logit and it beautifully boils down to

$$
\frac{\partial l}{\partial o_j} = \frac{\exp(o_j)}{\sum_k \exp(o_k)} - y_j = \hat{y}_j - y_j
$$

Prediction minus truth. That is all that reaches the weights.

This is the same shape as the regression gradient in chapter 01, where the error term was also prediction minus target. It is why the code barely changes between the chapters: there the error was `y_hat - y` and the weight gradient was `X.T @ error`; here the error is `y_hat - y_onehot` and the weight gradient is still `X.T @ error`. Same skeleton, different error term.

The rest of the derivation is worth doing once to build the math intuition. In practice this one line is the whole backward pass.

## Softmax revisited: LogSumExp

The naive path is:

1. get logits
2. exponentiate them for softmax
3. take the log of the true class probability for cross-entropy

Steps 2 and 3 undo each other, and between them sits a number computers handle badly. $\exp$ **overflows** when a logit is large, and **underflows** to zero when it is very negative, after which $\log 0$ is $-\infty$.

First fix: subtract the largest logit before exponentiating,

$$
\hat{y}_j = \frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})}, \qquad \bar{o} = \max_k o_k
$$

The answer does not change, because $e^{a-b} = e^a / e^b$ cancels top and bottom. Every exponent is now $\le 0$, so overflow is gone. Underflow is still a risk.

Second fix: never form the probability at all, go straight to its log,

$$
\log \hat{y}_j = o_j - \bar{o} - \log \sum_k \exp(o_k - \bar{o})
$$

This is **LogSumExp**, a simpler and safer form. The fragile $\exp$ never has to survive into the $\log$.

This is why frameworks fuse the two steps into a single operation: `nn.CrossEntropyLoss` in PyTorch, `from_logits=True` in Keras. We hand them logits, not probabilities. The from-scratch files keep the naive version on purpose, so the difference stays visible.

## Fashion-MNIST in practice

The dataset is 28×28 greyscale clothing images in 10 classes.

- Flatten each image from $28 \times 28$ into a 784-length vector. The model never knows it was a grid; that only arrives with convolutions later.
- $\mathbf{W}$ is $784 \times 10$ and $\mathbf{b}$ is length 10. Still no hidden layers, still the single layer from chapter 01.
- Softmax over the 10 logits, cross-entropy against the label.
- Training loop, batching and SGD are untouched.

A linear model reaches roughly 83% accuracy here. That is the baseline everything later in the book has to beat.

## Practice files

The same classifier written at four points on the ladder. See [levels of abstraction](../Index.md#levels-of-abstraction).

### [Softmax_Regression_Raw_Math.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/rough-notes/code/Softmax_Regression_Raw_Math.py){:target="_blank"}

Levels 1 and 2 together, in plain Python. No numpy, no torch, just nested loops over a small subset of the data. The hand-derived $\hat{y}_j - y_j$ gradient is plugged in directly. Slow, but nothing is hidden.

### [L1_From_Scratch.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/02-linear-neural-network-for-classification/L1_From_Scratch.py){:target="_blank"}

Level 2, vectorized with numpy. The same hand-derived gradient, now as matrix operations, with no autograd anywhere.

Implemented concepts:

1. Softmax and cross-entropy written out by hand
2. The $\hat{y} - y$ gradient applied directly
3. Mini-batch SGD on real image data

### [L2_PyTorch.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/02-linear-neural-network-for-classification/L2_PyTorch.py){:target="_blank"}

Level 3. `nn.Linear` holds the parameters, `nn.CrossEntropyLoss` fuses softmax and cross-entropy the LogSumExp way, `optim.SGD` applies the update. We still write the training loop.

Its docstring also keeps a note of **level 2.5**: raw autograd, where we create the tensors and step them ourselves and only the differentiation is handed over. Not implemented as its own file, but the book goes through it and it is worth remembering.

### [L3_High_Level.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/02-linear-neural-network-for-classification/L3_High_Level.py){:target="_blank"}

Level 4. Keras owns the loop. We describe the model, compile it, and call `fit()`.

## Generalization in classification

Theory and practice disagree here, and both are worth holding.

Statistical learning theory says a model with many parameters trained on little data will memorize it, and the bounds it produces demand enormous amounts of training data.

In practice that is often not what happens. Large models frequently generalize well on far less data than the bounds require. So the bounds are **pessimistic**: useful as intuition for what makes generalization harder, not as a budget for how much data to collect.

### How big should the test set be?

This is not about training. Training is already done. We are asking how confident we can be that the error measured on the test set is close to the error we will see on new data.

Uncertainty in an estimated rate shrinks as

$$
\frac{\sigma}{\sqrt{n}}
$$

Each test example is a pass/fail trial, so $\sigma = \sqrt{p(1-p)}$, which is largest at $p = 0.5$ and gives $\sigma = 0.5$.

For the measured error to sit within 1% of the truth one standard deviation of the time (~68%),

$$
\frac{0.5}{\sqrt{n}} = 0.01 \;\Rightarrow\; n = 2500
$$

For two standard deviations (~95%),

$$
\frac{2 \times 0.5}{\sqrt{n}} = 0.01 \;\Rightarrow\; n = 10000
$$

The relationship worth memorizing: **to halve the uncertainty we need four times the data.**

There is a second route through **Hoeffding's inequality**, which takes the tolerance (1%) and the confidence (95%) up front and returns a hard upper bound on $n$, around 15000 for the same requirement. It is more conservative because the central limit theorem *assumes* the errors settle into a normal distribution as $n$ grows, while Hoeffding proves the bound without that assumption.

### Test set reuse

The test set degrades every time we look at it, in two distinct ways.

- **Multiple hypothesis testing (false discovery).** Evaluate many candidate models on one test set and the best-looking one is partly just lucky on that particular set. The gaps we are reading may be noise, even though to our eyes the ranking looks meaningful.
- **Adaptive overfitting.** We see a disappointing score, go back, change the model, and try again. The test set has now leaked into our design decisions through us. It is no longer an honest estimate of anything.

The validation set exists for that trial and error. **The test set is sacred: touch it once, at the end.**

### Statistical learning theory and VC dimension

**VC dimension** is the largest number of points a model class can *shatter*, meaning separate correctly for every possible arrangement and every possible labelling of them.

A straight line in 2D can shatter 3 points. Add a fourth and there are labellings it cannot handle. In general a hyperplane in $d$ dimensions has VC dimension $d + 1$.

It gives a bound tying together the three things we care about,

$$
R[p, f] \le R_{\text{emp}}[\mathbf{X}, \mathbf{Y}, f] + c\sqrt{\frac{\text{VC} - \log \delta}{n}}
$$

True risk $\le$ measured risk $+$ a gap. The gap **grows** with model capacity (VC) and **shrinks** with more data ($n$), and $\delta$ sets the confidence.

What makes it interesting is that this is a purely mathematical statement about flexibility versus data, reached without running anything. It is the same lever weight decay pulls in chapter 01: restrain the model's flexibility and the generalization gap closes.

## Environment and distribution shift

Everything above assumes the world we test on is the world we trained on. When it is not, the model is not wrong; the distribution moved. Imagine walking into a mysterious world where dogs look like cats.

Which kind of shift it is comes from how we factor the joint distribution $p(x, y)$.

### Covariate shift

$$
p(x, y) = p(x)\, p(y \mid x)
$$

$p(y \mid x)$ holds, $p(x)$ changes. The *kind of input* we receive changes. Trained on photos, deployed on cartoons. A cat is still a cat; we are just seeing different cats.

### Label shift

$$
p(x, y) = p(y)\, p(x \mid y)
$$

$p(x \mid y)$ holds, $p(y)$ changes. The *frequency of each label* changes. Cancer looks the same as it always did, but in production it turns up far less often than the 10% we trained on.

### Concept shift

$p(y \mid x)$ itself changes. The same input now has a different correct answer. A model classifying "is this fashionable" goes stale on its own, with no change in the inputs at all. Geography does this as well as time.

## Correcting the shift

### Covariate shift correction

On the training data we minimize average loss,

$$
\text{minimize}_f \; \frac{1}{n}\sum_{i=1}^{n} l(f(\mathbf{x}_i), y_i)
$$

What we actually care about is loss over all real-world data,

$$
E_{p(\mathbf{x},y)}\big[l(f(\mathbf{x}), y)\big] = \iint l(f(\mathbf{x}), y)\, p(\mathbf{x}, y)\; d\mathbf{x}\, dy
$$

The weighting is what differs: the real-world version weighs each example by how likely we are to *meet* it out there, and the integral stands in for infinite data. So we reweight each training example by that ratio,

$$
\beta_i \stackrel{\text{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}
$$

$p$ is the real-world distribution and $q$ is the training one. Under-represented in training means $\beta_i > 1$, so we punish those errors a little harder, and the reverse for over-represented ones. Then we minimize **weighted empirical risk**,

$$
\text{minimize}_f \; \frac{1}{n}\sum_{i=1}^{n} \beta_i\, l(f(\mathbf{x}_i), y_i)
$$

Getting $\beta_i$ is the part that sounds strange the first time: **train a second, throwaway binary classifier** whose only job is to say whether an example came from the training pool or the real-world pool. With a logistic regression whose raw output is $h(\mathbf{x})$,

$$
P(z = 1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x}) + q(\mathbf{x})}
\quad\Longrightarrow\quad
\beta_i = \exp\big(h(\mathbf{x}_i)\big)
$$

Note what this needs: more real-world examples, but **no labels for them**. That is the whole appeal.

The catch: $\beta_i$ blows up when $q(\mathbf{x}_i)$ is near zero, which is exactly the case of a genuinely new kind of input that training never saw. Reweighting can only redistribute attention across what we already have. It cannot manufacture coverage that was never there.

### Label shift correction

Here $p(y)$ is unknown and we have no production labels.

1. Say training was balanced, 500 cats and 500 dogs.
2. Run the model over unlabelled production data. It predicts, say, 300 cats and 700 dogs. These are predictions rather than labels, so we do not trust the split yet.
3. Get the model's **confusion matrix** from labelled validation data, which is its known pattern of over- and under-calling each class.
4. Use it to nudge the production counts toward ground truth. The 300/700 might become 350/650.
5. That corrected split is our estimate of real-world $p(y)$. The weights are $p(y)/q(y)$, used in the loss the same way $\beta_i$ was.

### Concept shift correction

No principled method for this one. We keep training on fresh data on top of the existing weights and let the model drift along with the world. Old tech products losing popularity, news stories replaced by newer ones: these move gradually enough that continued training keeps up.

## A taxonomy of learning problems

Ordered by how much the environment talks back.

1. **Batch learning.** Train, deploy, walk away. The automatic door that only lets cats in.
2. **Online learning.** Labels arrive only after we predict, so we predict, observe, compute loss, and retrain. Stock price prediction.
3. **Bandits.** Online learning with a finite set of actions (arms) instead of a continuously parametrized model. Simpler, so stronger optimality guarantees are provable. A special case, not a separate setting.
4. **Control.** The environment has memory but no agenda; its response depends on what we did before. The coffee boiler is already warm from the last heating; a user will not read the same article twice. PID controllers are the standard tool.
5. **Reinforcement learning.** The environment has memory *and* its own agenda, so it may cooperate or compete. Chess, or other drivers reacting to an autonomous car.

The variable running through all five is whether the environment is **stationary or adaptive**. A strategy that works on a fixed environment can fail on one that reacts; an arbitrage opportunity disappears the moment it is exploited. How fast and how suddenly things change then picks the algorithm: slow drift means forcing estimates to change slowly, rare sudden jumps means allowing for them explicitly. That second case is concept shift again.

## Fairness, accountability, transparency

Predictions become real decisions, decisions change the world, and the changed world produces the next batch of training data. That is a **runaway feedback loop**, and it is why distribution shift is not purely a technical concern. A model trained on narrow data and deployed widely does not only measure the world; it starts shaping it.

---

## To revisit

- PID controllers: mentioned, not gone into.
- Derive Hoeffding's bound instead of quoting the number.
- Label shift correction with a confusion matrix, worked through in code.
