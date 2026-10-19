---
title: 03 Multilayer Perceptrons
parent: Permanent Notes
nav_order: 4
---

# 03 · Multilayer Perceptrons

Builds on [02 · Linear Neural Network for Classification](../02-linear-neural-network-for-classification/Notes.md). Book: D2L chapter 5.

## Summary

This chapter is basically where I started all this: [ANN_single_epoch_simulation.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/rough-notes/code/ANN_single_epoch_simulation.py){:target="_blank"} and the MNIST image classification problem.

In the last two chapters there was a single step to get the output; every input affected the output directly. MLP (multilayer perceptron) adds hidden layers in between, where every neuron of one layer directly affects every neuron of the next layer.

What this chapter covers, in short:

1. Layers only help if we add activation functions (ReLU, sigmoid, tanh). Without them, many layers collapse back into a single linear function.
2. Backprop now has to pass through every layer. The punishment gets routed back through the weights and the activation functions.
3. Going through many layers, gradients can vanish or explode. Random, well-scaled initialization (Xavier, He) keeps the values sane.
4. More layers means more flexibility, so generalization needs more care: early stopping and dropout.
5. Kaggle house prices puts it all together: data preprocessing, a better error measure using logs, and k-fold cross-validation to pick hyperparameters.

## Why layers?

With one hidden layer, the single step becomes two:

$$
\mathbf{h} = \phi(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})
$$

$$
\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h} + \mathbf{b}^{(2)}
$$

where $\mathbf{h}$ holds the hidden layer values (after the activation function $\phi$) and $\mathbf{o}$ is the output, same as in chapter 02. Every neuron of the previous layer directly affects every neuron in the next layer.

Why not keep a single layer with a huge number of neurons? They would all be directly affected by each input. Theoretically it can work, but the training process becomes more tedious.

With multiple layers, we can imagine every neuron looking for its "thing" in the previous layer. Pattern recognition becomes more intelligent when each neuron in the first layer starts to identify certain features and quirks (curves, lines, etc.). The next layer then has access to the high level conclusions made by the first layer, and makes further decisions on top of them.

## Activation functions

If no activation function is used, many layers at the end become a single function. Two linear layers simply merge into one:

$$
\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

with $\mathbf{W} = \mathbf{W}^{(2)}\mathbf{W}^{(1)}$ and $\mathbf{b} = \mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}$. Back to chapter 01's linear model, no matter how many layers we stack.

By adding an activation function we add non-linearity. Imagine a flat line/plane trying to find patterns in a 2d/3d graph vs a flexible line/plane trying to find patterns in the same space. Activation functions give this flexibility to move around and fit the data.

### ReLU

ReLU (rectified linear unit):

$$
\text{ReLU}(x) = \max(x, 0)
$$

It is straightforward. We let the value pass as it is if it is positive, and make it zero if it is negative.

Its gradient becomes either 0 or 1, so the gradient passes through without being shrunk: either the entire gradient or nothing. This helps against the [vanishing gradients](#vanishing-gradients) problem.

On the other hand, when the output is zero, that neuron doesn't learn anything and doesn't pass anything back to the previous layer. If that happens for every input, the neuron is stuck for good. This is called the dead ReLU problem.

Another option is parametrized ReLU (pReLU):

$$
\text{pReLU}(x) = \max(0, x) + a \cdot \min(0, x)
$$

We let negatives come in, but scaled by $a$. So some gradient still flows for negative values and the neuron can't die completely. $a$ is not a fixed constant; it is learned along with the weights, that is the "parametrized" part.

### Sigmoid

$$
\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}
$$

Sigmoid converts any value into a value between 0 and 1. As values get bigger it leans towards 1, and as they get smaller it goes towards 0. With input 0, sigmoid gives 0.5. (Same squashing function as in [00 · Basics](../00-basics/Notes.md#model-building-blocks).)

Rather than just saying if a neuron fired or not, sigmoid tells in what intensity it fired. kinda. It is actually the older one of the two; ReLU took over the hidden layers later, mostly because of the drawback below.

Its derivative:

$$
\frac{d}{dx}\,\text{sigmoid}(x) = \text{sigmoid}(x)\,\big(1 - \text{sigmoid}(x)\big)
$$

At $x = 0$ this is $0.5 \times 0.5 = 0.25$, its biggest value. Far from 0 it is almost 0. The drawback is vanishing gradients: as we pass through multiple layers, the gradient keeps getting smaller and smaller ([more below](#vanishing-gradients)).

### Tanh

$$
\tanh(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
$$

Similar to sigmoid, but the output lies between −1 and 1. Nothing much was mentioned about the good, the bad and the use cases of tanh.

There are more activation functions out there, GELU and more. More later.

## Implementation

We use `nn.Sequential` to just say how many layers there are and the width of each layer, and PyTorch gives us that without us worrying about the inner architecture:

```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), nn.ReLU(),  # hidden layer + activation
                    nn.Linear(256, 10))              # output layer
```

Same Fashion-MNIST setup as chapter 02, with one hidden layer of 256 neurons added in between. This is level 3 on the [levels of abstraction](../Index.md#levels-of-abstraction) ladder.

## Backprop, revisited

Backprop came again in the book. I derived $\partial J / \partial \mathbf{W}^{(2)}$ and $\partial J / \partial \mathbf{W}^{(1)}$ on paper, and this is how I understand it. The book leaves out the biases here:

$$
\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x}, \qquad \mathbf{h} = \phi(\mathbf{z}), \qquad \mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}
$$

$J$ is the loss $L$ plus the weight decay part from [chapter 01](../01-linear-neural-network-for-regression/Notes.md), $J = L + \frac{\lambda}{2}\left(\lVert\mathbf{W}^{(1)}\rVert^2 + \lVert\mathbf{W}^{(2)}\rVert^2\right)$, so every weight gradient below also gets $+\lambda\mathbf{W}$ on top. Leaving that part aside:

**1. Start at the output.** $\partial J / \partial \mathbf{o}$ is the important one that gets carried all the way down. It is the derivative of the loss with respect to the prediction made:

- MSE: $\frac{2}{n}(\hat{y} - y)$ ([chapter 01](../01-linear-neural-network-for-regression/Notes.md))
- softmax + cross-entropy: simply $\hat{\mathbf{y}} - \mathbf{y}$ ([chapter 02](../02-linear-neural-network-for-classification/Notes.md#the-gradient-that-matters))

**2. Output layer weights.** We multiply it with the $\mathbf{h}$ values that came into this layer ($\mathbf{h}$ = values after the activation function of the previous layer). Every weight gets its own $h$, so each weight is punished in proportion to what passed through it:

$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}} = \frac{\partial J}{\partial \mathbf{o}}\,\mathbf{h}^\top
$$

**3. Hidden layer.** Walk the path backward, $\mathbf{o} \to \mathbf{h} \to \mathbf{z} \to \mathbf{W}^{(1)}$, one multiplication per step.

*Through $\mathbf{W}^{(2)}$.* $\mathbf{W}^{(2)}$ is like a gatekeeper. In the forward pass it decided how much of each $h$ reaches the output, so on the way back the same strictness decides who gets more punishment. It scales the value, shrinking or growing it depending on the weight:

$$
\frac{\partial J}{\partial \mathbf{h}} = \mathbf{W}^{(2)\top}\,\frac{\partial J}{\partial \mathbf{o}}
$$

*Through the activation function.* For ReLU it becomes simply 0 or 1, for sigmoid $h(1-h)$:

$$
\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{h}} \odot \phi'(\mathbf{z})
$$

where $\odot$ means element by element.

*Into $\mathbf{W}^{(1)}$.* Same as the output layer, now with $\mathbf{x}$ instead of $\mathbf{h}$. Again the punishment is in proportion to who contributed what:

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}}\,\mathbf{x}^\top
$$

The order matters once these are matrices: we follow the path backward, so the activation step comes before $\mathbf{x}$. For a single number it wouldn't matter.

**Why the transpose?** It is not only there to match shapes. Going forward, $\mathbf{W}^{(2)}$ sends each $h_j$ to every output $o_k$. Going back, each $h_j$ has to collect punishment from every output it fed:

$$
\frac{\partial J}{\partial h_j} = \sum_k W^{(2)}_{kj}\,\frac{\partial J}{\partial o_k}
$$

That sum runs down a column of $\mathbf{W}^{(2)}$, which is a row of $\mathbf{W}^{(2)\top}$. So the transpose is the same connections used in the reverse direction. Same idea for $\mathbf{h}^\top$ and $\mathbf{x}^\top$: weight $W_{ij}$ connects input $j$ to neuron $i$, so its punishment is (punishment of neuron $i$) $\times$ (input $j$), and $\frac{\partial J}{\partial \mathbf{z}}\,\mathbf{x}^\top$ builds that whole grid at once.

This is the same thing I did by hand for a tiny 2-2-1 sigmoid network in [ANN_single_epoch_simulation.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/rough-notes/code/ANN_single_epoch_simulation.py){:target="_blank"}, where `error_signal_from_output_layer` gets routed back through w5 and w6.

## Numerical stability and initialization

This section is about keeping the values of weights and biases sane.

### Vanishing gradients

Sigmoid comes with this risk. Although it is considered a thresholding function similar to the human brain, it can make gradients zero. Reason? Sigmoid squishes values between 0 and 1, and its derivative is at most 0.25 ([sigmoid](#sigmoid)). Backprop multiplies one such derivative per layer, so the gradient keeps shrinking as it goes through multiple layers. After 10 layers it is at most $0.25^{10}$, about one millionth.

ReLU on the other hand is better at this, as we allow the entire value to pass through. The only concern with ReLU is the [dead ReLU](#relu) situation, where neurons that became zero are useless. There are some mitigation methods for that; pReLU is one.

### Exploding gradients

Similar mirror case, where values keep growing until the numbers overflow (become inf or NaN). Same kind of overflow problem as exp in [chapter 02's LogSumExp](../02-linear-neural-network-for-classification/Notes.md#softmax-revisited-logsumexp).

### Breaking the symmetry

It is important to assign random values to the weights at the beginning, to make the model expressive. If all weights started the same, every hidden neuron would compute the same thing and get the same punishment in backprop. They would stay identical forever, and the whole layer would act like a single neuron. Biases can start at zero; in chapter 02's code $\mathbf{b}$ started at zeros and only $\mathbf{W}$ was random.

### Initialization: Xavier and He

If we use ML frameworks, this is all handled by the libraries and we don't have to worry about it. For the sake of keeping a note, I understood it at a high level. Might need to dive deeper in future if required.

So far I just picked $\sigma = 0.01$ for the weights by hand (chapter 02 code). Xavier and He are principled ways to pick it.

**Xavier.** We stop at each layer and ask: given that this layer has $n_\text{in}$ inputs and $n_\text{out}$ outputs, what variance should I use while drawing weights from a normal or uniform distribution, so that the output variance stays roughly the same as the input variance?

Forward pass, with $\sigma^2$ the variance of the weights and $\gamma^2$ the variance of the inputs:

$$
\text{Var}(o_i) = n_\text{in}\,\sigma^2\gamma^2
$$

We want $\text{Var}(o_i) \approx \gamma^2$, so $n_\text{in}\,\sigma^2 \approx 1$, which gives $\sigma^2 \approx 1/n_\text{in}$.

Asking the same question for backward propagation: a hidden neuron receives gradients from $n_\text{out}$ outputs, which gives $\sigma^2 \approx 1/n_\text{out}$. We can't satisfy both at once, so Xavier takes the compromise:

$$
\sigma^2 = \frac{2}{n_\text{in} + n_\text{out}}
$$

Example: $n_\text{in} = 100$ and $n_\text{out} = 50$ give $\sigma^2 = 2/150 \approx 0.0133$, so $\sigma \approx 0.115$. Instead of standard deviation 1 we use around 0.115. Huge difference.

Uniform version: for $W \sim U(-a, a)$ the variance is $a^2/3$. Setting $a^2/3 = 2/(n_\text{in} + n_\text{out})$ gives

$$
W \sim U\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}},\ \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)
$$

Same idea, different distribution. Xavier goes well with tanh and sigmoid.

**He.** Same purpose, for ReLU. ReLU throws away the negative half, so roughly half of the signal is gone at every layer (this is ReLU's normal zeroing, not the dead ReLU problem). To compensate, we increase the variance, doubling the forward-pass value $1/n_\text{in}$:

$$
\sigma^2 = \frac{2}{n_\text{in}}, \qquad \sigma = \sqrt{\frac{2}{n_\text{in}}}
$$

## Generalization in deep learning

Some repetition of points from previous chapters ([generalization in chapter 02](../02-linear-neural-network-for-classification/Notes.md#generalization-in-classification)) and some general talk:

1. **Optimization and generalization are different tasks.** Optimization is about fitting the training data. Generalization is about understanding the underlying patterns, the ones that stay reliable for unknown data as well.
2. **Inductive biases matter.** Inductive bias is a model's preference towards certain configurations of parameters over others. It can come from the architecture selection, the optimization algorithm, initialization etc. We can imagine models having their own way of achieving things. We may or may not always be able to control it.
3. **Some of it is still a mystery.** Some parts of the training process, and the question of why models generalize well, are still mysterious at some level. They are smart brute force approaches that do impressive work, but we can't say we have 100% theoretical proof that they are the best approach. (Same feeling as chapter 02, where the theory was pessimistic but practice generalized better.)

### Inspiration from nonparametrics

> "Functionally, neural networks look like parametric models. But thinking of them as nonparametric models can sometimes be a more reliable source of intuition."

(Parametric: a fixed set of weights that summarizes the data. Nonparametric: the model leans on the training examples themselves and grows with the data.)

Some parallels were drawn between neural nets and kernels. Kernels are yet to learn, so this point stays open till finished. I spent some time understanding kernels and SVM (support vector machines) and parked it after getting a high level idea, as there is much more to it.

It feels related to the expressivity idea I noted outside the book ([Rough Note Random](../../rough-notes/Rough_Note_Random.md)): when it is hard to draw a dividing line in n-d space, project the data into (n+1)-d space and dividing becomes possible.

### Early stopping

A technique to stop early when the validation loss has reached its bottom, after which the model starts fitting to noise rather than actual patterns. It is the point where more optimization stops helping generalization.

## Dropout

A regularization method (same family as weight decay in chapter 01) where we randomly disable neurons during training. The model becomes more robust with this strategy, and empirically it is proven to generalize better. It also reduces co-dependency by making neurons independently confident, less dependent on relying on a few other neurons.

With dropout probability $p$, each hidden value $h$ becomes

$$
h' = \begin{cases} 0 & \text{with probability } p \\ \dfrac{h}{1-p} & \text{otherwise} \end{cases}
$$

The surviving neuron activations are bumped up by dividing them by $1 - p$. If the same input is processed thousands of times, we expect the average activation of any given neuron to stay the same: we suppress it sometimes, and when it is left open we bump it up:

$$
E[h'] = p \cdot 0 + (1-p)\cdot\frac{h}{1-p} = h
$$

That is also why nothing needs to change at test time, when dropout is off and every neuron is used: the average was already kept the same during training.

Scratch and concise implementations are covered in the book, very straightforward.

## Kaggle: predicting house prices

### Data preprocessing

**Standardization.** Numerical data are rescaled with respect to their distance to the mean and the standard deviation:

$$
x \leftarrow \frac{x - \mu}{\sigma}
$$

It makes sense with this example: there are two columns, one with data like 10, 23, 12, 9, the other with 56000, 30000, 20000. We can't pass them to the model as they are; the big column would dominate the weighted sum and the gradients, and one learning rate can't suit both scales. After this step both become series with mean 0 and standard deviation 1.

This step feels important and worth keeping. Concept wise it feels rich and might come back somewhere else too.

**Missing values.** Then we fill null values with zero. Since the mean is now 0, that means we are placing the missing ones at the mean.

**Categorical values** are converted into one-hot encodings by creating a separate column for each value (same one-hot idea as the labels in chapter 02). `dummy_na=True` also adds a column for "missing":

```python
features = pd.get_dummies(features, dummy_na=True)
```

### Error measure

The common instinct when defining a loss function is $(y - \hat{y})^2$. However there are issues with that.

**Issue 1:** a house price difference of 100k is not the same for a 200k house and a 5M house, so taking the proportion is important. We could use $(y - \hat{y})/y$. Even better, use logs, because $\log a - \log b = \log(a/b)$. So this is enough:

$$
\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log y_i - \log \hat{y}_i\right)^2}
$$

**Issue 2:** the predicted price can become negative.

To solve issue 2 we go one step ahead: transform the labels into $\log(\text{labels})$ upfront. Teach the model to predict log values and use them directly in the loss function. Converting back with $\exp$ always gives a positive price (same reason exp was used in [softmax](../02-linear-neural-network-for-classification/Notes.md#softmax)).

### K-fold cross-validation and model selection

K-fold cross-validation is used to decide on the hyperparameters of the model. For example with $k = 5$: take one configuration of hyperparameters and run a loop 5 times. Each time, train a fresh new model with 4 folds and use the last fold as the validation set. Collect the losses and take the average.

```
         fold 1   fold 2   fold 3   fold 4   fold 5
run 1     val     train    train    train    train
run 2    train     val     train    train    train
 ...
run 5    train    train    train    train     val
```

Repeat this with a different hyperparameter configuration, gather its average loss and compare.

Beware that trying too many different configs brings model selection bias, where we just get lucky with some config and can't find the one that generalizes on production data. Same problem as [test set reuse](../02-linear-neural-network-for-classification/Notes.md#test-set-reuse) in chapter 02.

Why not one single model that we train on 80% of all the labeled data? Because some model might give a low validation error just by chance. What if the model learned some quirks and patterns that luckily apply to the selected 20% validation set? The concept of [distribution shift](../02-linear-neural-network-for-classification/Notes.md#environment-and-distribution-shift) connects here, and so does [test set size](../02-linear-neural-network-for-classification/Notes.md#how-big-should-the-test-set-be): a small validation set gives a noisy error.

In cross-validation we keep rotating the validation set and take the average from multiple models, making the error more robust and less prone to distribution shift when the real test data comes.

### Final model and submitting on Kaggle

The last step was preparing predictions and submitting them on Kaggle.

Once the hyperparameters are decided, we can take that config and train freshly on all the training data. However, the D2L book used the same 5 models to predict the house prices for the test data (averaging their predictions). It can work but doesn't seem like a standard industry practice.

## Practice files

### [Kaggle_House_Prices.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/03-multilayer-perceptrons/Kaggle%20House%20Price%20Prediction/Kaggle_House_Prices.py){:target="_blank"}

Level 3 on the [levels of abstraction](../Index.md#levels-of-abstraction) ladder: `nn` + `optim`, training loop written by me. A boilerplate baseline for the [Kaggle contest](https://www.kaggle.com/c/house-prices-advanced-regression-techniques){:target="_blank"}, to play around with later.

Implemented concepts:

1. Preprocessing from above: standardization, nulls filled with 0 (the mean), one-hot with `dummy_na=True`
2. Labels turned into log prices upfront, so the loss is the log RMSE and `exp` gives back positive prices
3. 5-fold cross-validation printing train / validation log RMSE per fold and the average
4. Final model retrained on all training data (not the book's average of 5 fold models), writing `submission.csv` ready to upload

All the knobs are in `CONFIG` at the top: `hidden` (empty = plain linear regression, `[256]` adds a hidden layer), `dropout`, `weight_decay`, `lr`, `epochs`, `batch_size`, `k`. The baseline (linear, Adam, 100 epochs) gives an average validation log RMSE of about 0.15.

---

## To revisit

- Tanh: good, bad and use cases (the book didn't say much).
- GELU and the other activation functions.
- Other mitigation methods for dead ReLU, beyond pReLU.
- Xavier and He: understood at a high level, dive deeper if ever needed.
- Kernels and SVM, and how neural nets connect to nonparametric models.
- Final model: is the standard practice to retrain on all data, or to average the k fold models?
