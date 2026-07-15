import numpy as np
np.random.seed(3)

examples = [([1,1],6), ([2,0],5), ([0,1],4), ([3,2],13), ([1,2],9), ([2,3],14)]
X = np.array([e[0] for e in examples], dtype=float)
y = np.array([e[1] for e in examples], dtype=float)

# --- 1. NORMALIZE INPUTS ---
print("=== STEP 1: CALCULATING NORMALIZATION VALUES ===")
mean = X.mean(axis=0)
std  = X.std(axis=0)
Xn   = (X - mean) / std

print(f"Raw Input A Mean    : {mean[0]:.2f} | Spread (Std): {std[0]:.2f}")
print(f"Raw Input B Mean    : {mean[1]:.2f} | Spread (Std): {std[1]:.2f}")
print("\nFirst 3 Raw Inputs vs Normalized Inputs:")
for i in range(3):
    print(f"  Raw: {X[i]} -> Normalized: [{Xn[i][0]:.4f}, {Xn[i][1]:.4f}] (Centered near 0)")
print("=" * 75 + "\n")

# Random start
w    = np.random.randn(2) * 0.5
bias = np.random.randn() * 0.5
step = 0.2

print("=== STEP 2: STARTING TRAINING (TRACKING FIRST 3 STEPS) ===")
print(f"STARTING WEIGHTS: w1_norm={w[0]:.4f}, w2_norm={w[1]:.4f}, bias_norm={bias:.4f}\n")

total_steps_tracked = 0

for ep in range(1, 13):
    for a, target in zip(Xn, y):
        guess = w @ a + bias
        error = target - guess
        
        # Track the first 3 individual updates
        if total_steps_tracked < 3:
            total_steps_tracked += 1
            print(f"[Epoch {ep}, Step {total_steps_tracked}]")
            print(f"  -> Normalized Input vector 'a' : [{a[0]:.4f}, {a[1]:.4f}]")
            print(f"  -> Target Output               : {target}")
            print(f"  -> Dot Product Math            : ({w[0]:.4f} * {a[0]:.4f}) + ({w[1]:.4f} * {a[1]:.4f}) + {bias:.4f}")
            print(f"  -> AI Guessed                  : {guess:.4f} (Error: {error:.4f})")
            
            # Show the calculation for the updates
            w1_nudge = step * a[0] * error
            w2_nudge = step * a[1] * error
            b_nudge  = step * error
            print(f"  -> Nudging w1_norm by          : {step} * {a[0]:.4f} * {error:.4f} = {w1_nudge:+.4f}")
            print(f"  -> Nudging w2_norm by          : {step} * {a[1]:.4f} * {error:.4f} = {w2_nudge:+.4f}")
            print(f"  -> Nudging bias_norm by        : {step} * {error:.4f} = {b_nudge:+.4f}")
            
            # Apply updates
            w    += step * a * error
            bias += step * error
            print(f"  -> NEW NORMALIZED VALUES       : w1_norm={w[0]:.4f}, w2_norm={w[1]:.4f}, bias_norm={bias:.4f}")
            print("-" * 75)
        else:
            # Update silently for the rest of the 12 epochs
            w    += step * a * error
            bias += step * error
            
    # At the end of every epoch, print the macro progress
    worst = np.max(np.abs((Xn @ w + bias) - y))
    if ep == 1:
        print("\n... Switching to Macro Epoch view (watching the worst error drop) ...\n")
        print("Epoch |   w1'     w2'    bias'  | Worst error remaining across all data")
        print("-" * 75)
    print("%4d  | %6.3f  %6.3f  %6.3f |  %.4f" % (ep, w[0], w[1], bias, worst))

print("=" * 75 + "\n")
print("=== STEP 3: CONVERTING BACK TO RAW WEIGHTS ===")
# --- convert the normalized weights back to your familiar 2, 3, 1 ---
w_raw = w / std
b_raw = bias - np.sum(w * mean / std)

print("Final Normalized Weights : w1'=%.3f  w2'=%.3f  bias'=%.3f" % (w[0], w[1], bias))
print(f"Conversion Formula Matrix: w_raw = w / std  |  b_raw = bias - sum(w * mean / std)")
print("Converted to Raw Weights : w1=%.2f   w2=%.2f   bias=%.2f   <- THE HIDDEN RULE DISCOVERED!" % (w_raw[0], w_raw[1], b_raw))
print("-" * 75)

# Final Validation Test
a_val, b_val = 4, 1
print(f"TEST ON NEW UNSEEN INPUT [{a_val},{b_val}]:")
print(f"  -> Raw Formula Math: ({w_raw[0]:.2f} * {a_val}) + ({w_raw[1]:.2f} * {b_val}) + {b_raw:.2f}")
print(f"  -> AI Final Prediction: {w_raw @ [a_val, b_val] + b_raw:.2f} (True Answer: 12)")
