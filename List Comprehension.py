#Advanced patterns of comprehension for Machine Learning
# 1 - The Flattening Pattern for Nsted Loops
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# flat = []
# for row in matrix:
#     for num in row:
#         flat.append(num)
# print(flat)
#faltening pattern using list comprehension
flat = [num for row in matrix for num in row]
print(flat)
