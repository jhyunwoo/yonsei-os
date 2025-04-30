from random import randint

matrix_size = 1024
head = int(input("Enter the process size: "))
matrix_size = int(matrix_size // head**(1/3))

with open("test/multi/input1.txt", "w") as f:
    f.write(f"{head}\n")

    for k in range(head):
        for l in range(3):
            f.write(f"{matrix_size} {matrix_size}\n")
            for i in range(matrix_size):
                row = ' '.join(str(randint(0, 100)) for _ in range(matrix_size))
                f.write(row + '\n')