import numpy as np
from scipy.interpolate import BarycentricInterpolator

def u_exact(x):
    return 0.5 * x * (1 - x)

def chebyshev(N):
    return np.array([np.cos(np.pi * j / N) for j in range(N+1)])

#approximates u''
def D1(N) :
    x = np.cos(np.pi * np.arange(N + 1) / N )
    c = np.insert(np.ones(N-1),[0,N-1], [2,2])
    D_N = np.empty(shape = (N+1, N+1))
    for j in range(N + 1) :
        for l in range(N + 1) :
            if j == 0 and l == 0 :
                D_N[j,l] = (2 * N ** 2 + 1) / 6
            if j == N and l == N :
                D_N[j,l] = - (2 * N ** 2 + 1) / 6
            if j == l and j < N and j > 0 :
                D_N[j,l] = - x[j] / (2 * (1 - x[j] ** 2))
            if j != l :
                D_N[j,l] =  c[j]/c[l] * ((-1) ** (j + l)) / (x[j] - x[l])
    return D_N

def D2(N):
    D = D1(N)
    return D @ D

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


