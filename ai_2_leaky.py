import numpy as np

# LEAKY ReLU squash: negatives are kept, just shrunk (x0.1).
def squash(z):
    return np.where(z > 0, z, 0.1 * z)

x = np.array([1, 0])                 # SAME input as file 1

# SAME weights as file 1
W1 = np.array([[ 0.5, -0.4],
               [-0.3,  0.8],
               [ 0.9,  0.1]])
b1 = np.array([0.0, 0.2, -0.1])

layer1_scores = W1 @ x + b1
layer1_out    = squash(layer1_scores)

W2 = np.array([[0.6, -0.2, 0.7]])
b2 = np.array([0.05])
final_guess = W2 @ layer1_out + b2

print("raw scores   :", layer1_scores)
print("after squash :", layer1_out, "  <- negative neuron kept (shrunk)")
print("final guess  :", final_guess)
