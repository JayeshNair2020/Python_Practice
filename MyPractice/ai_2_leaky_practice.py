import numpy as np

def sqaush(z):
    return np.where(z > 0, z, 0.1 * z) # how where works: it checks the condition (z > 0) for each element in the array z. If the condition is True, it keeps the original value; if False, it multiplies the value by 0.1, effectively shrinking negative values while keeping positive values unchanged.

W1 = np.array([[ 0.5, -0.4],
                [-0.3,  0.8]])
b1 = np.array([0.0, 0.2])
X1 = np.array([1, 0])                 # SAME input as file 1
layer1_scores = W1 @ X1 + b1
layer1_out    = sqaush(layer1_scores)

W2 = np.array([[0.6, -0.2]])
b2 = np.array([0.05])
final_guess = W2 @ layer1_out + b2

print("raw scores   :", layer1_scores)
print("after squash :", layer1_out, "  <- negative neuron kept (shrunk)")
print("final guess  :", final_guess)