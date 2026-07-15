import numpy as np
np.random.seed(3)

# In real AI, NOBODY types the importance numbers. The AI starts with RANDOM
# weights and learns the right ones from EXAMPLES, using the exact nudge you know:
#     weight = weight + step * input * error
#
# Here the hidden rule (which the AI does NOT know) is:  score = 2*a + 3*b + 1
# We only show it examples. Watch it discover 2, 3, 1 on its own.

examples = [([1,1],6), ([2,0],5), ([0,1],4), ([3,2],13), ([1,2],9), ([2,3],14)] 

# start with RANDOM importance numbers -- we did NOT choose these
w1, w2, bias = np.random.randn(3) * 0.5
print("random start weights :  w1=%.2f  w2=%.2f  bias=%.2f" % (w1, w2, bias))

step = 0.01
for run in range(2000):
    for (a, b), target in examples:
        guess = w1*a + w2*b + bias      # the guess
        error = target - guess          # how wrong
        w1   += step * a * error        # nudge each weight by its share
        w2   += step * b * error
        bias += step * 1 * error

print("learned weights      :  w1=%.2f  w2=%.2f  bias=%.2f   <- it FOUND the rule" % (w1, w2, bias))

# test on a brand-new input it never saw:
a, b = 4, 1                              # true answer = 2*4 + 3*1 + 1 = 12
print("test [4,1] -> guess %.2f   (true answer 12)" % (w1*a + w2*b + bias))
