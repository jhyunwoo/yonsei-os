from random import randint

matrix_size = int(input("Enter the matrix size: "))

with open("test/attention/input1.txt", "w") as f:
    for k in range(3):
        f.write(f"{matrix_size} {matrix_size}\n")
        for i in range(matrix_size):
            row = " ".join(str(randint(0, 100)) for _ in range(matrix_size))
            f.write(row + "\n")