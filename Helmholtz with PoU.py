import numpy as np
import matplotlib.pyplot as plt

domain = [0,10]

lam = 30

def u_exact(x):
    return np.sin(2*x) + np.cos(5*x)

def f(x):
    return 26*np.sin(2*x) + 5*np.cos(5*x)

def centers_radii(M):
    return [0,8], 4

def normalize_coordinate(x,xn,r_n):
    return (x-xn)/r_n

#Returns POU value of x in center xn

#Returns POU value of x in center xn
def psi(x, xn, radii):
    xtilde = normalize_coordinate(x, xn, radii)

    if -5/4 <= xtilde < -3/4:
        return (1 + np.sin(2 * np.pi * xtilde)) / 2
    elif -3/4 <= xtilde < 3/4:
        return 1
    elif 3/4 <= xtilde < 5/4:
        return (1 - np.sin(2 * np.pi * xtilde)) / 2
    else:
        return 0

#Returns value of the first derivative of the POU in x
def first_derivative_psi(x, xn, radii):
    xtilde = normalize_coordinate(x, xn, radii)
    scale = 1 / radii

    if -5 / 4 <= xtilde < -3 / 4:
        return (np.pi * np.cos(2 * np.pi * xtilde)) * scale
    elif -3 / 4 <= xtilde < 3 / 4:
        return 0
    elif 3 / 4 <= xtilde < 5 / 4:
        return (-np.pi * np.cos(2 * np.pi * xtilde)) * scale
    else:
        return 0


#Returns value of the second derivative of the POU in x
def second_derivative_psi(x, xn, r_n):
    xtilde = normalize_coordinate(x, xn, r_n)
    scale = (1 / r_n) ** 2

    if -5 / 4 <= xtilde < -3 / 4:
        return (-2 * np.pi ** 2 * np.sin(2 * np.pi * xtilde)) * scale
    elif -3 / 4 <= xtilde < 3 / 4:
        return 0
    elif 3 / 4 <= xtilde < 5 / 4:
        return (2 * np.pi ** 2 * np.sin(2 * np.pi * xtilde)) * scale
    else:
        return 0


#Returns a list of random vector '(weight, bias)' for each random feature function
#weight: Jn weights in [-R, R]^d
#bias: Jn biases in [-R, R]
def generate_feature_vectors(Jn, R):
    weights = np.random.uniform(-5, 5, size=Jn)
    biases = np.random.uniform(-np.pi, np.pi, size=Jn)
    return [[weights[i], biases[i]] for i in range(Jn)]

#Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector):
    return np.cos(x * feature_vector[0] + feature_vector[1])

#Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector):
    return feature_vector[0]*(-np.sin(x * feature_vector[0] + feature_vector[1]))

#Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector):
    return feature_vector[0]**2*(-np.cos(x * feature_vector[0] + feature_vector[1]))

#sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)

#not being used currently
def collocation_points_boundary():
    return [domain[0], domain[1]]

M = 100
Jn = 20

#calculate values
centers, radii = centers_radii(M)
Mp = len(centers)
print(centers)

#For each center, we generate a list of Jn random features vectors (weight,bias)
feature_vectors_list = [generate_feature_vectors(Jn, radii) for i in range(Mp)]

#To compute the matrices
def P(x, xn, feature_vector, r):
    return (psi(x, xn, r) * second_derivative_feature(x, feature_vector)
            + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector)
            + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector)
            + lam * psi(x,xn,r) * feature_function(x,feature_vector))

#Initialize matrices to zero
A = np.zeros((Mp * Jn, Mp * Jn))
B = np.zeros(Mp * Jn)

#Choose collocation points
collocation_points = collocation_points_interior(M)

#Compute matrice entries
for N in range(0,Mp):
    for J in range(0,Jn):
        for n in range(0, Mp ):
            for j in range(0, Jn):
                total = 0
                for x in collocation_points:
                    P_nj = P(x, centers[n], feature_vectors_list[n][j], radii)
                    P_NJ = P(x, centers[N], feature_vectors_list[N][J], radii)
                    total += 2 * P_nj * P_NJ
                A[N * Jn + J, n * Jn + j] += total
                total = 0
                for x in domain:
                    total += 2* psi(x,centers[n], radii) * feature_function(x,feature_vectors_list[n][j]) * psi(x,centers[N], radii) * feature_function(x,feature_vectors_list[N][J])
                A[N * Jn + J, n * Jn + j] += total
        total = 0
        for xi in collocation_points:
            P_NJ = P(xi, centers[N], feature_vectors_list[N][J], radii)
            total += 2 * f(xi) * P_NJ
        for xb in domain:
            total += 2 * u_exact(xb) * psi(xb, centers[N], radii) * feature_function(xb, feature_vectors_list[N][J])
        B[N * Jn + J] = total

#Solve to find optimal u_values
U = np.linalg.solve(A, B)

#Calculate approximate solution using u_values
def approximate_solution(x):
    total= 0
    for n in range(0,Mp):
        pou = psi(x, centers[n], radii)
        for j in range(0,Jn):
            unj = U[n * Jn + j]
            feature_value = feature_function(x, feature_vectors_list[n][j])
            total += unj* feature_value * pou
    return total

x_values = np.linspace(domain[0], domain[1], 300)
aproximation = [approximate_solution(x) for x in x_values]
exact = [u_exact(x) for x in x_values]

#plot result
plt.plot(x_values, exact)
plt.plot(x_values, aproximation, '--')
max_diff = np.max(np.abs(np.array(aproximation) - np.array(exact)))
plt.text(0.02, 0.98, f'Max error: {max_diff:.6f}',
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()
