import numpy as np
from scipy.interpolate import BarycentricInterpolator

def u_exact(x):
    return 0.5 * x * (1 - x)

def chebyshev(N):
    return np.array([np.cos(np.pi * j / N) for j in range(N+1)])

#approximates u''
def D2(N):
    x = chebyshev(N)
    c = np.array([2] + [1]*(N-1) + [2])
    D2_matrix = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for l in range(N + 1):
            if j != l and j != 0 and j != N:
                D2_matrix[j, l] = ((-1) ** (j + l) / c[j] * (x[j]**2 + x[j]*x[l] - 2)
                            / ((1 - x[j]**2) * (x[j] - x[l])**2))
            elif j == l and 1 <= j <= N - 1:
                D2_matrix[j, j] = -((N**2 - 1) * (1 - x[j]**2) + 3) / (3 * (1 - x[j]**2)**2)
            elif j == 0 and l != 0:
                D2_matrix[j, l] = (2/3 * (-1)**l / c[l] * ((2*N**2 + 1) * (1 - x[l]) - 6)
                            / (1 - x[l])**2)
            elif j == N and l != N:
                D2_matrix[j, l] = (2/3 * (-1)**(l + N) / c[l] * ((2*N**2 + 1) * (1 + x[l]) - 6)
                            / (1 + x[l])**2)
    D2_matrix[0, 0] = (N**4 - 1) / 15
    D2_matrix[N, N] = (N**4 - 1) / 15
    return D2_matrix

N = 10
chebyshev_nodes = chebyshev(N)
A = 4.0 * D2(N)
B = np.array([0] + [1 for i in range(N-1)] + [0])

#Enforce boundary
A[0, :] = 0
A[0, 0] = 1
A[N, :] = 0
A[N, N] = 1

U = np.linalg.solve(-A, B)

def approximate_solution(x, cheby_nodes, U):
    x = 2 * x - 1
    polynomial = BarycentricInterpolator(cheby_nodes, U)
    return polynomial(x)

points = np.linspace(0, 1, 300)
approximation = [approximate_solution(x, chebyshev_nodes, U) for x in points]
exact = [u_exact(x) for x in points]
errors = abs(np.array(approximation) - np.array(exact))
print(f"Max error: {np.max(errors):.2e}")


