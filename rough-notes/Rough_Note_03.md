---
title: Rough Note 03
parent: Rough Notes
nav_order: 3
---

Linear Neural Networks for Classification

Softmax - needed when we need probability destribution.
we use e**x to handle negatives and it also amplifies the differences between numbers.
we normalize values by taking proportional values of total sum of exponential functions.

cross-entropy formula: 

- y (log y hat)

two ways to derive it
1. taking sum of real values multiplied by log of expected value. why log -> it gives spread out spectrum for values input values between 0 and 1 giving higher value for lower inputs and slowly reducing it to 0 as we reach 1. simple case when batch size = 1.

2. maximum likelihood estimation. this is a way of probability where we are combining logs of multiple predicted outputs. see this as way to find loss when batch size > 1.

further intuition development of entropy and cross entropy.

Entropy is quantification of chaos. in this instance, it says how much the probablity is accumulated at one place or how much it is scattered. Ex. [1, 0, 0] has 0 entropy while [1/3,1/3,1/3] has highest entropy as there is no certainty at all.

oj (log oj) would give entropy of single element out of a vector. we do multiplication with oj to give weights to each surprisal. doing sum of of weighted entropis of all elements would give average entropy.

for cross-entropy, we replace multiplication part. we take real probability (coming from labels) to multiply with entropy of predicted probability.

if we add softmax function in cross-entropy formula while getting delta l / delta oj, it beautifully boils down to y hat(j) - y(j). So I suppose this should be the only thing that matters in practical case. Rest seems important to develop math intuition.

- so summarizing this cross-entropy part
- formula is simple however going deeper into where it comes from gives better idea about the math in action.
- 2 ways to see it.
- first, maximum likelihood estimation - we want to maximize the likelihood of label Y when feaure X are given. As they are mutually independent scenarios we do multiplication -> for many multiplication, it would keep going near zero -> so we introduce log. log(ab) = log a + log b. -> log (yi | xi) is y hat j -> multiplying them with y (one-hot encoding) gives cross entropy formula.

- second, entropy and information theory - log value of a predicted probability gives surprisal value of any given prediction. the more it is near 1, we are less suprised. but if an event with 1% chance occurs, we are more surpised. log does exactly that -> multiplying them with their own prediction and summing them up gives us weighted average. multiplication because although some events might have high entropy but it doesn't occur all the time, so multiplying it with its initial prediction value gives its part in total entropy. and that becomes entropy function. -> In the case of classification model where we know real labels and their probability, we can do this multiplication with actual labels and that gives up cross-entropy formula.


-----------------------
some practicals start

Fashion-MNIST dataset classification

- Softmax Regression Implementation from Scratch

below two are new things, rest remains more or less similar.

def softmax(X):
X_exp = torch.exp(X)
partition = X_exp.sum(1, keepdims=True)
return X_exp / partition # The broadcasting mechanism is applied here

def cross_entropy(y_hat, y):
return-torch.log(y_hat[list(range(len(y_hat))), y]).mean()
cross_entropy(y_hat, y)

by combining things we can create classifier for Fashion-MNIST image dataset.
- we flatten input data from 28 x 28 to 784 x 1
- just like linear regression we don't have any hidden layers yet. The new additions however are softmax function to get probability distribution out of predicted numbers and we have cross-entropy loss calculation instead of MSE.
- training and back propagation steps remain same.

- Softmax concise implementation

nn.sequential - we can define layers of process while defining neural network

softmax revisited (interesting one)
LogSumExp

The fundamental knowledge says to
1. get logits
2. find softmax (use of exponential function)
3. put sofmax value of real label into log (use of log function; cross entropy)

use of eponential function is problematic in computers; it can underflow or overflow.

Better approach?
use max(all logit values) - o(J) inside exponential function. mathematically response doesn't change due to rule e**(a-b) = e**a / e**b.

this will save us from overflow. underflow is still a risk

what then?

putting softmax function inside log with use of rule explained above will give LogSumExp formula.
it is a simpler form and safe.

Pytorch internally uses this method.


- Generalization in Classification

summary part re-emphasizes importance of generalization and adds another point.
When we have a complex model (many parameters) with relatively small amount of test data, our initial instinct says it will overfit because model will memorize all train data. However, it is not like that. Although theory says that way, in practice it is au contraire good at generalizing.

So, the point I understood right now is
there are some statistical methods to find generalization beforehand, but it gives pessimistic insights. For example, you will need too many train data.
However, when we do actual training, more often we don't need that many training data.
So, I suppose the chapter will further talk about the industry principles and all.

- 4.6.1 The Test Set

This section addresses ways to predetermine number of test data. Reason? to be confident and aware about our error on any new data that may come in future.
Note that we are not talking about training data. training is already done and we are talking about testing and measuring errors.

First way is with Central Limit Theorem.

It is based on this logic
uncertainty increases proportional to sigma / underroot n ; where sigma is standard deviation and n is number of data.

uncertainty will be percentage of allowed variance (let's say 0.01 aka 1%)
standard deviation will be 0.5 (comes from bernoulli's rule sigma = p(1-p) and p in worst case of uncertainty is 0.5)

if given above 2 values, n will come out as 2500.
It means if around 2500 random set are ran through classifier and recorded deviation from mean error and let's say we put them on a distribution graph, ~68% (1 standard deviation) of cases will obey our 1% variance condition.

If we want to make it happening for 95% of the cases (2 standard deviation), same formula will give n as 10000.

One important relationship out of this is, if we want uncertainty to reduce by half, our dataset size should increase 4 times.

There is another way to take this estimate.
It is using Hoeffding formula. It is a formula that takes bounds of allowed variance (1%) and intended distribution spread to be covered (95%) upfront and gives upper bound of n with surety.

In this case it would give 15000. (It gave 10000 for the same requirement)

The main difference is that central limit theorem assumes that as n increases, the cumulative variance from mean follows a normal distribution.
The Hoeffding formula uses math to directy calculate the value of n without assumption.