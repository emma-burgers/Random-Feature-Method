import numpy as np
from scipy.interpolate import BarycentricInterpolator

L = 10

def u_exact(x):
    return np.sin(2*x) + np.cos(5*x)

def f(x):
    return 26*np.sin(2*x) + 5*np.cos(5*x)

def chebyshev(N):
    return np.array([np.cos(np.pi * j / N) for j in range(N+1)])

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

N = 25
cheby_nodes = chebyshev(N)

x_phys = L/2 * (cheby_nodes + 1)

scale = (2/L)**2
A = scale * D2(N) + 30 * np.eye(N+1)
B = f(x_phys)

# Enforce boundary conditions
A[N, :] = 0;  A[N, N] = 1;  B[N] = u_exact(0)
A[0, :] = 0;  A[0, 0] = 1;  B[0] = u_exact(L)

U = np.linalg.solve(A, B)

def approximate_solution(x_in, cheby_nodes, U):
    t = 2*x_in/L - 1
    poly = BarycentricInterpolator(cheby_nodes, U)
    return poly(t)

points = np.linspace(0, L, 300)
approximation = np.array([approximate_solution(x, cheby_nodes, U) for x in points])
exact = u_exact(points)
errors = np.abs(approximation - exact)
print(f"Max error: {np.max(errors)}")