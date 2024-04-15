import numpy as np


def LU(A):
    n = len(A)
    L = [[0.0 if i != j else 1.0 for j in range(n)] for i in range(n)]
    U = [row[:] for row in A]
    P = list(range(n))

    for k in range(n):
        max_index = max(range(k, n), key=lambda i: abs(U[i][k]))
        if k != max_index:
            U[k], U[max_index] = U[max_index], U[k]
            P[k], P[max_index] = P[max_index], P[k]
            if k > 0:
                L[k][:k], L[max_index][:k] = L[max_index][:k], L[k][:k]

        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U, P

def solve_system(L, U, b):
    n = len(A)
    # L, U, P = LU(A)
    # Lz = b
    z = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * z[j]
        z[i] = (b[i] - s) / L[i][i]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (z[i] - s) / U[i][i]

    return x


def det(A):
    n = len(A)
    L, U, P = LU(A)
    result = 1
    for i in range(n):
        result *= U[i][i]

    # for i in range(n):
    #     result *= L[i][i]

    return result


def transpose(A):
    A_T = [[] for _ in range(len(A[0]))]
    for i in range(len(A_T)):
        A_T[i].extend([A[j][i] for j in range(len(A))])
    return A_T


def inverse(A):
    n = len(A)
    # E^-1
    E = [[float(i == j) for i in range(n)] for j in range(n)]

    inversed = []


    L, U, _= LU(A)

    for column in E:
        x = solve_system(L, U, column)
        inversed.append(x)

    return transpose(inversed)

def permutation_matrix(P):
    n = len(P)
    P_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        P_matrix[i][P[i]] = 1
    return P_matrix


if __name__ == "__main__":
    A = [[-5, -1, -3, -1], 
         [-2, 0, 8, -4], 
         [-7, -2, 2, -2], 
         [2, -4, -4, 4]]
    
    b = [18, -12, 6, -12]


    # A = [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]]

    # A = [
    #     [23994, 340923, 239492, -12],
    #     [1239, 2304, 20, -20],
    #     [234, 2342, 23, 0],
    #     [0, 1, 0, 4],
    # ]
    L, U, P = LU(A)
    print("L = ")
    print(np.array(L))
    
    print("U = ")
    print(np.array(U))

    
    print("P = ")
    P = np.array(permutation_matrix(P))
    print(P)
    


    print("LU = ")
    print(np.array(L) @ np.array(U))
    # print(np.linalg.inv(P) * np.array(L) @ np.array(U))





    x = solve_system(L, U, b)

    print(f"Ax = b, x = {x} \n")
    print(f"detA = {det(A)} \n")
    # print(np.linalg.det(np.array(A)))

    inversed = inverse(A)

    print("A inversed = ")
    print(np.array(inversed))