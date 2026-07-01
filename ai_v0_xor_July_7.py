import numpy as np

# Your leaky squash: keep positives as-is, shrink negatives (don't kill them).
def squash(z):
    return np.where(z > 0, z, 0.1 * z)

# ---------------- ONE input goes in ----------------
x = np.array([1, 0])          # two numbers (like a movie's action & comedy)

# ---------------- LAYER 1: three little scoring machines ----------------
# Each ROW is one neuron's importance numbers.
W1 = np.array([[ 0.5, -0.4],   # neuron 1's importance for the 2 inputs
               [-0.3,  0.8],   # neuron 2
               [ 0.9,  0.1]])  # neuron 3
b1 = np.array([0.0, 0.2, -0.1])  # each neuron's mood (bias)

layer1_scores = W1 @ x + b1          # run all 3 machines -> 3 scores
layer1_out    = squash(layer1_scores) # squash each score

# ---------------- LAYER 2: one final scoring machine ----------------
# It looks at the 3 outputs from layer 1, so it has 3 importance numbers.
W2 = np.array([[0.6, -0.2, 0.7]])    # one neuron, 3 weights
b2 = np.array([0.05])                # its mood

final_score = W2 @ layer1_out + b2   # the AI's final guess

print("layer 1 raw scores :", layer1_scores)
print("after the squash   :", layer1_out)
print("final guess        :", final_score)