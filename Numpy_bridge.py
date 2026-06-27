#NUmpy is where python meets numerical Ai that is math
#Numpy can run all operations on whole array at once called vectorization
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
c = a + b
d = a*b
e = a**2
print(c) # [11 22 33 44 55] (element-wise addition)
print(d) # [10 40 90 160 250] (element-wise multiplication)
print(e) # [1 4 9 16 25] (element-wise square)
print(np.mean(a)) # 3.0 (mean of array a)
print(np.std(a)) # 1.4142135623730951 (standard deviation of array a)
print(np.max(a)) # 5 (maximum value in array a)
print(np.min(a)) # 1 (minimum value in array a)

# Every single operation in your code scales linearly based on the size of the array:
# c = a + b O(n) (n additions)
# d = a * b O(n) (n multiplications)
# e = a ** 2 O(n) (n exponentiations)