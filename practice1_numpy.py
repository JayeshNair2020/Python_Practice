# PRACTICE BEFORE STAGE 1
# Write one function summarize(nums) that returns a dict with the count, sum, mean, min, and max of a list — using comprehensions and built-ins, no manual
# loops where avoidable. Then rewrite the mean using NumPy.
import numpy as np
def summarize(nums):
    if not nums:
        return {
            'count': 0,
            'sum': 0,
            'mean': None,
            'min': None,
            'max': None
        } #handle empty list otherwise will get zero division error when calculating mean
    return {
        'count': len(nums),
        'sum': sum(nums),
        'mean': np.mean(nums),
        'min': min(nums),
        'max': max(nums)
    }
list = [1, 2, 3, 4, 5]
result = summarize(list)
print(result) # {'count': 5, 'sum': 15, 'mean': 3.0, 'min': 1, 'max': 5}