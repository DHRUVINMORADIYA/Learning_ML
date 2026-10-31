---
title: Rough Note 05
parent: Rough Notes
nav_order: 5
---

Chapter 6: Builder's Guide

This chapter feels like focused on more practicals and less on new concepts and theories.


6.1 Layers and Modules
this entire segment covers sequential module, custom module and even a mix implementation approach.
- when we are straight passing output of one layer to the next, we can simply use ready maid nn.sequential.
- If ever there is a case where we want to add our own logic between layers, we can use custom module.

below is an example code showing all possibilities with custom module

class NestMLP(nn.Module):
def __init__(self):
super().__init__()
self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
nn.LazyLinear(32), nn.ReLU())
self.linear = nn.LazyLinear(16)
def forward(self, X):
return self.linear(self.net(X))
chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)

Fixedhiddenmlp is another custom built module.

6.2 Parameter Management
1. this section shows how to access weights and biases of each layer with simple call.
2. Introduced to Parameter as a pytorch datatype.
3. Introduced to concept of shared(tied) parameters.

in below code layer 'shared' is used two times. It is the same tensor with same values. In backpropagation, pytorch handles gradients by combining contribution of shared weights from both places.

# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
shared, nn.ReLU(),
shared, nn.ReLU(),
nn.LazyLinear(1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])

6.3 Parameter Initialization
- Pytorch methods for parameters initialization - custom, inbuilt and hybrid

6.4 Lazy Initialization
- In pytorch modules weights are initilized when first inference is run, so we don't have to define input parameters count for each layer in general. It is calculated at runtime.

6.5 Custom Layers
- shows method to create custom layer with or without parameters. straightforward.

6.6 File I/O
- It is possible to store and load tensors to/from disk.
- It is possible to store and load model parameters to/from disk. architecture of model is still required to be recreated or cloned in code. Only parameters are stored and restored.