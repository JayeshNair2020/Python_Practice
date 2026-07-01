import numpy as np
np.random.seed(1)

# ---------- THE PROBLEM: XOR ----------
# output should be 1 only when the two inputs DIFFER.
# This is famous: a single scoring machine CANNOT solve it.
# You need layers + a squash. Which you now understand.
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# ---------- YOUR AI, built only from things you know ----------
# layer 1: W1 @ x + b1  ->  squash        (hidden detectors)
# layer 2: W2 @ h + b2                     (final score)
H = 8
W1 = np.random.randn(2, H) * 0.7;  b1 = np.zeros((1, H))
W2 = np.random.randn(H, 1) * 0.7;  b2 = np.zeros((1, 1))
lr = 0.1

def squash(z):        return np.where(z > 0, z, 0.1*z)   # your leaky squash
def squash_slope(z):  return np.where(z > 0, 1.0, 0.1)

for step in range(4001):
    # ----- forward: guess -----
    z1 = X @ W1 + b1
    h  = squash(z1)
    out = h @ W2 + b2

    # ----- compare: error -----
    err = out - y
    loss = np.mean(err**2)

    # ----- nudge: learn (backprop, exactly your rule chained) -----
    d_out = 2*err / len(X)
    dW2 = h.T @ d_out;              db2 = d_out.sum(0, keepdims=True)
    d_h = d_out @ W2.T
    d_z1 = d_h * squash_slope(z1)
    dW1 = X.T @ d_z1;              db1 = d_z1.sum(0, keepdims=True)
    W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1

    if step in (0, 200, 1000, 4000):
        print(f"step {step:>4}   error {loss:.4f}")

# ---------- did it learn? ----------
print("\nfinal predictions:")
for xi, yi in zip(X, y):
    guess = squash(xi @ W1 + b1) @ W2 + b2
    print(f"  input {xi.astype(int)}  ->  {guess[0][0]:.2f}   (want {int(yi[0])})")
