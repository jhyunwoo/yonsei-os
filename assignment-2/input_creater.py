# import random
#
# N = 1024  # 행과 열 크기
# FILENAME = "input.txt"
#
# def print_matrix(rows, cols):
#     print(f"{rows} {cols}")
#     for _ in range(rows):
#         print(" ".join(str(random.randint(0, 9)) for _ in range(cols)))
#
# with open(FILENAME, "w") as f:
#     # redirect print to file
#     import sys
#     sys.stdout = f
#     print_matrix(N, N)  # Q
#     print_matrix(N, N)  # K
#     print_matrix(N, N)  # V

import random

def print_matrix(rows, cols):
    print(f"{rows} {cols}")
    for _ in range(rows):
        print(" ".join(str(random.randint(0, 9)) for _ in range(cols)))

num_heads = 10
matrix_size = 1024  # Q/K/V 모두 동일 크기

print(num_heads)

for _ in range(num_heads):
    print_matrix(matrix_size, matrix_size)  # Q
    print_matrix(matrix_size, matrix_size)  # K
    print_matrix(matrix_size, matrix_size)  # V