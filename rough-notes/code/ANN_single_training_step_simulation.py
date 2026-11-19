import math

def sigmoid(x):
    """Squashes any number into the range (0, 1) - lets the network output a probability."""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sigmoid_output):
    """Slope of sigmoid, expressed in terms of its own output (standard shortcut)."""
    return sigmoid_output * (1 - sigmoid_output)

# =============================================================================
# NETWORK MAP - where each variable below sits in the network
# =============================================================================
#
#   INPUT              HIDDEN                        OUTPUT
#
#   x1 ----w1----\
#                 >--(+b1)--> z1 --sigmoid--> a1 ----w5----\
#   x2 ----w2----/                                           \
#                                                              >--(+b3)--> z3 --sigmoid--> a3  <-- compare to y
#   x1 ----w3----\                                            /
#                 >--(+b2)--> z2 --sigmoid--> a2 ----w6----/
#   x2 ----w4----/
#
#   x1, x2 = the two inputs                y  = the ideal/target answer
#   w1, w2 = weights into hidden neuron 1 (z1/a1)
#   w3, w4 = weights into hidden neuron 2 (z2/a2)
#   w5, w6 = weights into the output neuron (z3/a3)
#   b1     = bias of hidden neuron 1        b2 = bias of hidden neuron 2        b3 = bias of the output neuron
#   z      = weighted sum going into a neuron (before activation)
#   a      = that neuron's output (after sigmoid activation)
# =============================================================================

x1 = 2
x2 = 3
y  = 1
lr = 0.1

w1 = 0.1
w2 = 0.2
w3 = 0.4
w4 = 0.5
w5 = 0.7
w6 = 0.8

b1 = 0.3
b2 = 0.6
b3 = 0.9

# Forward Propagation

z1 = x1*w1 + x2*w2 + b1
#  = 2*0.1 + 3*0.2 + 0.3
#  = 1.1

a1 = round(sigmoid(z1),4) # z1 = 1.1
#  = 0.7503

print("Hidden neuron 1 -> z1 (weighted sum + bias):", z1, " | a1 (after sigmoid):", a1)

z2 = x1*w3 + x2*w4 + b2
#  = 2*0.4 + 3*0.5 + 0.6
#  = 2.9

a2 = round(sigmoid(z2),4) # z2 = 2.9
#  = 0.9478

print("Hidden neuron 2 -> z2 (weighted sum + bias):", z2, " | a2 (after sigmoid):", a2)

z3 = a1*w5 + a2*w6 + b3
#  = 0.7503*0.7 + 0.9478*0.8 + 0.9
#  = 2.18345

a3 = round(sigmoid(z3),4) # z2 = 2.9
#  = 0.8988

print("Output neuron  -> z3 (weighted sum + bias):", z3, " | a3 (prediction, after sigmoid):", a3)


# Loss Calculation (Squared Error)

loss = ((y - a3) ** 2) / 2
#    = ((1 - 0.8988) ** 2) / 2
#    = 0.0051

loss = round(loss,4)

print("Loss -> squared error between target y and prediction a3:", loss)


# Back Propagation (Sequence --> w5 -> w6 -> b3 -> b1 -> b2 -> w1 -> w2 -> w3 -> w4)

# w5

# w5 only reaches the loss through this path:  w5 --> z3 --> a3 --> loss
# So, by the chain rule:
# ∂loss/∂w5 = ∂loss/∂a3 * ∂a3/∂z3 * ∂z3/∂w5

# Piece 1: ∂loss/∂a3 -- how loss changes as the prediction a3 changes
# loss = (y - a3)^2 / 2  -->  derivative w.r.t a3 = -(y - a3) = (a3 - y)
d_loss_a3 = round(a3 - y, 4)
#         = 0.8988 - 1
#         = -0.1012

# Piece 2: ∂a3/∂z3 -- how a3 changes as z3 changes (slope of sigmoid at a3)
# a3 = sigmoid(z3)  -->  derivative = sigmoid(z3) * (1 - sigmoid(z3)) = a3 * (1 - a3)
d_a3_z3 = round(a3 * (1 - a3), 4)
#       = 0.8988 * (1 - 0.8988)
#       = 0.091

# Piece 3: ∂z3/∂w5 -- how z3 changes as w5 changes
# z3 = a1*w5 + a2*w6 + b3  -->  derivative w.r.t w5 = a1 (everything else is constant)
d_z3_w5 = a1
#       = 0.7503

# Chain rule: multiply the three pieces together to get the full effect of w5 on loss
d_loss_w5 = round(d_loss_a3 * d_a3_z3 * d_z3_w5, 4)
#         = -0.1012 * 0.091 * 0.7503
#         = -0.0069

print("w5 gradient -> d(loss)/d(a3):", d_loss_a3, " | d(a3)/d(z3):", d_a3_z3, " | d(z3)/d(w5):", d_z3_w5, " | d(loss)/d(w5):", d_loss_w5)

# Update w5: take a small step AGAINST the gradient (that's the direction that reduces loss)
w5_new = round(w5 - lr * d_loss_w5, 4)
#      = 0.7 - 0.1 * (-0.0069)
#      = 0.7007

print("w5 update  -> old w5:", w5, " | new w5:", w5_new)


# w6

# ∂loss/∂w6 = ∂loss/∂a3 * ∂a3/∂z3 * ∂z3/∂w6 = error_signal_from_output_layer * a2
error_signal_from_output_layer = round(d_loss_a3 * d_a3_z3, 4)
#                               = -0.0092

d_z3_w6 = a2
#       = 0.9478

d_loss_w6 = round(error_signal_from_output_layer * d_z3_w6, 4)
#         = -0.0092 * 0.9478
#         = -0.0087

print("w6 gradient -> error_signal_from_output_layer:", error_signal_from_output_layer, " | d(z3)/d(w6):", d_z3_w6, " | d(loss)/d(w6):", d_loss_w6)

w6_new = round(w6 - lr * d_loss_w6, 4)
#      = 0.8 - 0.1 * (-0.0087)
#      = 0.8009

print("w6 update  -> old w6:", w6, " | new w6:", w6_new)


# b3

# z3 = a1*w5 + a2*w6 + b3  -->  d(z3)/d(b3) = 1
# d(loss)/d(b3) = error_signal_from_output_layer * 1
d_loss_b3 = error_signal_from_output_layer
#         = -0.0092

print("b3 gradient -> d(loss)/d(b3):", d_loss_b3)

b3_new = round(b3 - lr * d_loss_b3, 4)
#      = 0.9 - 0.1 * (-0.0092)
#      = 0.9009

print("b3 update  -> old b3:", b3, " | new b3:", b3_new)


# b1

# b1 reaches the loss through this path:  b1 --> z1 --> a1 --> z3 --> a3 --> loss
# So, by the chain rule:
# ∂loss/∂b1 = ∂loss/∂a3 * ∂a3/∂z3 * ∂z3/∂a1 * ∂a1/∂z1 * ∂z1/∂b1
# The first two pieces are already known (error_signal_from_output_layer).
# ∂z3/∂a1 = w5, so that error signal is routed back through w5.

# ∂loss/∂a1 = error_signal_from_output_layer * w5  (error routed back through w5)
d_loss_a1 = round(error_signal_from_output_layer * w5, 4)
#         = -0.0092 * 0.7
#         = -0.0064

# ∂a1/∂z1 = a1 * (1 - a1)
d_a1_z1 = round(a1 * (1 - a1), 4)
#       = 0.7503 * (1 - 0.7503)
#       = 0.1873

# error_signal_from_hidden_neuron_1 = ∂loss/∂a1 * ∂a1/∂z1
error_signal_from_hidden_neuron_1 = round(d_loss_a1 * d_a1_z1, 4)
#                                  = -0.0064 * 0.1873
#                                  = -0.0012

# z1 = x1*w1 + x2*w2 + b1  -->  ∂z1/∂b1 = 1
d_loss_b1 = error_signal_from_hidden_neuron_1
#         = -0.0012

print("b1 gradient -> error_signal_from_hidden_neuron_1:", error_signal_from_hidden_neuron_1, " | d(loss)/d(b1):", d_loss_b1)

b1_new = round(b1 - lr * d_loss_b1, 4)
#      = 0.3 - 0.1 * (-0.0012)
#      = 0.3001

print("b1 update  -> old b1:", b1, " | new b1:", b1_new)


# b2

# b2 reaches the loss through:  b2 --> z2 --> a2 --> z3 --> a3 --> loss
# ∂loss/∂b2 = error_signal_from_output_layer * ∂z3/∂a2 * ∂a2/∂z2 * ∂z2/∂b2 = error_signal_from_output_layer * w6 * a2*(1-a2) * 1

d_loss_a2 = round(error_signal_from_output_layer * w6, 4)
#         = -0.0092 * 0.8
#         = -0.0074

d_a2_z2 = round(a2 * (1 - a2), 4)
#       = 0.9478 * (1 - 0.9478)
#       = 0.0495

error_signal_from_hidden_neuron_2 = round(d_loss_a2 * d_a2_z2, 4)
#                                  = -0.0074 * 0.0495
#                                  = -0.0004

d_loss_b2 = error_signal_from_hidden_neuron_2
#         = -0.0004

print("b2 gradient -> error_signal_from_hidden_neuron_2:", error_signal_from_hidden_neuron_2, " | d(loss)/d(b2):", d_loss_b2)

b2_new = round(b2 - lr * d_loss_b2, 6)
#      = 0.6 - 0.1 * (-0.0004)
#      = 0.60004

print("b2 update  -> old b2:", b2, " | new b2:", b2_new)


# w1

# w1 reaches the loss through this path:  w1 --> z1 --> a1 --> z3 --> a3 --> loss
# So, by the chain rule:
# ∂loss/∂w1 = ∂loss/∂a3 * ∂a3/∂z3 * ∂z3/∂a1 * ∂a1/∂z1 * ∂z1/∂w1
# The first four pieces are already known -- that's error_signal_from_hidden_neuron_1.
# z1 = x1*w1 + x2*w2 + b1  -->  ∂z1/∂w1 = x1

d_loss_w1 = round(error_signal_from_hidden_neuron_1 * x1, 4)
#         = -0.0012 * 2
#         = -0.0024

print("w1 gradient -> error_signal_from_hidden_neuron_1:", error_signal_from_hidden_neuron_1, " | d(z1)/d(w1):", x1, " | d(loss)/d(w1):", d_loss_w1)

w1_new = round(w1 - lr * d_loss_w1, 4)
#      = 0.1 - 0.1 * (-0.0024)
#      = 0.1002

print("w1 update  -> old w1:", w1, " | new w1:", w1_new)


# w2

# ∂loss/∂w2 = error_signal_from_hidden_neuron_1 * ∂z1/∂w2 = error_signal_from_hidden_neuron_1 * x2
d_loss_w2 = round(error_signal_from_hidden_neuron_1 * x2, 4)
#         = -0.0012 * 3
#         = -0.0036

print("w2 gradient -> error_signal_from_hidden_neuron_1:", error_signal_from_hidden_neuron_1, " | d(z1)/d(w2):", x2, " | d(loss)/d(w2):", d_loss_w2)

w2_new = round(w2 - lr * d_loss_w2, 4)
#      = 0.2 - 0.1 * (-0.0036)
#      = 0.2004

print("w2 update  -> old w2:", w2, " | new w2:", w2_new)


# w3

# ∂loss/∂w3 = error_signal_from_hidden_neuron_2 * ∂z2/∂w3 = error_signal_from_hidden_neuron_2 * x1
d_loss_w3 = round(error_signal_from_hidden_neuron_2 * x1, 4)
#         = -0.0004 * 2
#         = -0.0008

print("w3 gradient -> error_signal_from_hidden_neuron_2:", error_signal_from_hidden_neuron_2, " | d(z2)/d(w3):", x1, " | d(loss)/d(w3):", d_loss_w3)

w3_new = round(w3 - lr * d_loss_w3, 4)
#      = 0.4 - 0.1 * (-0.0008)
#      = 0.4001

print("w3 update  -> old w3:", w3, " | new w3:", w3_new)


# w4

# ∂loss/∂w4 = error_signal_from_hidden_neuron_2 * ∂z2/∂w4 = error_signal_from_hidden_neuron_2 * x2
d_loss_w4 = round(error_signal_from_hidden_neuron_2 * x2, 4)
#         = -0.0004 * 3
#         = -0.0012

print("w4 gradient -> error_signal_from_hidden_neuron_2:", error_signal_from_hidden_neuron_2, " | d(z2)/d(w4):", x2, " | d(loss)/d(w4):", d_loss_w4)

w4_new = round(w4 - lr * d_loss_w4, 4)
#      = 0.5 - 0.1 * (-0.0012)
#      = 0.5001

print("w4 update  -> old w4:", w4, " | new w4:", w4_new)