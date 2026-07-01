import numpy as np

# NORMAL ReLU squash: negatives become a hard 0.
def squash(z):
    return np.maximum(0, z)

x = np.array([1, 0])                 # one input

# LAYER 1: three scoring machines (each ROW is one neuron)
W1 = np.array([[ 0.5, -0.4],
               [-0.3,  0.8],
               [ 0.9,  0.1]])
b1 = np.array([0.0, 0.2, -0.1])

layer1_scores = W1 @ x + b1
layer1_out    = squash(layer1_scores)

# LAYER 2: one final machine
W2 = np.array([[0.6, -0.2, 0.7]])
b2 = np.array([0.05])
final_guess = W2 @ layer1_out + b2

print("raw scores   :", layer1_scores)
print("after squash :", layer1_out, "  <- negative neuron killed to 0")
print("final guess  :", final_guess)
