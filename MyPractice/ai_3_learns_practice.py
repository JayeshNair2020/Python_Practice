import numpy as np
np.random.seed(3)
# In real AI, NOBODY types the importance numbers. The AI starts with RANDOM
# weights and learns the right ones from EXAMPLES, using the exact nudge you know:
#     weight = weight + step * input * error
#
# Here the hidden rule (which the AI does NOT know) is:  score = 2*a + 3*b + 1
# We only show it examples. Watch it discover 2, 3, 1 on its own.
examples = [([1,1],6), ([2,0],5), ([0,1],4), ([3,2],13), ([1,2],9), ([2,3],14)] #example 2(1) + 3(1) + 1 = 6, example 2(2) + 3(0) + 1 = 5, example 2(0) + 3(1) + 1 = 4, example 2(3) + 3(2) + 1 = 13, example 2(1) + 3(2) + 1 = 9, example 2(2) + 3(3) + 1 = 14

# Start with random weights
w1, w2, bias = np.random.randn(3) * 0.5
print(f"STARTING VALUES: w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}\n")
print("-" * 75)

step = 0.01
total_steps_tracked = 0

for run in range(2000):
    for (a, b), target in examples:
        # 1. Calculate the current guess
        guess = w1*a + w2*b + bias
        error = target - guess
        
        # 2. Track and print the exact math for the first 5 steps
        if total_steps_tracked < 5:
            print(f"INPUT: a={a}, b={b} | TARGET: {target}")
            print(f"   -> Current Formula: ({w1:.4f} * {a}) + ({w2:.4f} * {b}) + {bias:.4f}")
            print(f"   -> AI Guessed     : {guess:.4f} (Error: {error:.4f})")
            
            # Show the exact nudge calculation
            print(f"   -> Nudge w1 by    : {step} * {a} * {error:.4f} = +{step * a * error:.4f}")
            print(f"   -> Nudge w2 by    : {step} * {b} * {error:.4f} = +{step * b * error:.4f}")
            print(f"   -> Nudge bias by  : {step} * 1 * {error:.4f} = +{step * 1 * error:.4f}")
            
            # Apply the updates
            w1   += step * a * error
            w2   += step * b * error
            bias += step * 1 * error
            
            print(f"   -> NEW VALUES     : w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}")
            print("-" * 75)
            total_steps_tracked += 1
        else:
            # For the rest of the 2,000 runs, just update silently
            w1   += step * a * error
            w2   += step * b * error
            bias += step * 1 * error

print("\n... Training silently for the remaining loops ...\n")
print(f"FINAL LEARNED WEIGHTS: w1={w1:.2f}, w2={w2:.2f}, bias={bias:.2f}")

# Final Test
a, b = 4, 1
print(f"TEST ON NEW INPUT [{a},{b}] -> AI Guess: {w1*a + w2*b + bias:.2f} (True Answer: 12)")
