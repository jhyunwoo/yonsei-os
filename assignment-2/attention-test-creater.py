import random

N = 1024  # 행과 열 크기
FILENAME = "../optimization/att-input.txt"

def print_matrix(rows, cols):
    print(f"{rows} {cols}")
    for _ in range(rows):
        print(" ".join(str(random.randint(0, 9)) for _ in range(cols)))

with open(FILENAME, "w") as f:
    # redirect print to file
    import sys
    sys.stdout = f
    print_matrix(N, N)  # Q
    print_matrix(N, N)  # K
    print_matrix(N, N)  # V