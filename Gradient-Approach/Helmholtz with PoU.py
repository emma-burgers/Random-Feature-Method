import numpy as np
import matplotlib.pyplot as plt

domain = [0,6]

lam = 30

def u_exact(x):
    return np.sin(2*x) + np.cos(5*x)

def f(x):
    return 26*np.sin(2*x) + 5*np.cos(5*x)

def centers_radii(N):
    centers = np.linspace(domain[0], domain[1], num=N)
    return centers, (centers[1] - centers[0]) / 2

def normalize_coordinate(x,xn,r_n):
    return (x-xn)/r_n

#Returns POU value of x in center xn
def psi(x, xn, radii):
    x_norm = normalize_coordinate(x, xn, radii)

    if -5/4 <= x_norm < -3/4:
        return (1 + np.sin(2 * np.pi * x_norm)) / 2
    elif -3/4 <= x_norm < 3/4:
        return 1
    elif 3/4 <= x_norm < 5/4:
        return (1 - np.sin(2 * np.pi * x_norm)) / 2
    else:
        return 0

#Returns value of the first derivative of the POU in x
def first_derivative_psi(x, xn, radii):
    x_norm = normalize_coordinate(x, xn, radii)

    if -5 / 4 <= x_norm < -3 / 4:
        return (np.pi * np.cos(2 * np.pi * x_norm)) * (1 / radii)
    elif -3 / 4 <= x_norm < 3 / 4:
        return 0
    elif 3 / 4 <= x_norm < 5 / 4:
        return (-np.pi * np.cos(2 * np.pi * x_norm)) * (1 / radii)
    else:
        return 0


#Returns value of the second derivative of the POU in x
def second_derivative_psi(x, xn, r_n):
    x_norm = normalize_coordinate(x, xn, r_n)
    if -5 / 4 <= x_norm < -3 / 4:
        return (-2 * np.pi ** 2 * np.sin(2 * np.pi * x_norm)) * ((1 / r_n) ** 2)
    elif -3 / 4 <= x_norm < 3 / 4:
        return 0
    elif 3 / 4 <= x_norm < 5 / 4:
        return (2 * np.pi ** 2 * np.sin(2 * np.pi * x_norm)) * ((1 / r_n) ** 2)
    else:
        return 0

#Returns a list of m parameter vectors '[(weight, bias)]'
def generate_feature_vectors(m):
    weights = np.random.uniform(-8, 8, size=m)
    biases = np.random.uniform(-np.pi, np.pi, size=m)
    return [[weights[i], biases[i]] for i in range(m)]


# Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector, xn, rn):
    x_norm = normalize_coordinate(x, xn, rn)
    return np.cos(x_norm * feature_vector[0] + feature_vector[1])

# Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector, xn, rn):
    x_norm = normalize_coordinate(x, xn, rn)
    return feature_vector[0] * (-np.sin(x_norm * feature_vector[0] + feature_vector[1])) * (1 / rn)

# Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector, xn, rn):
    x_norm = normalize_coordinate(x, xn, rn)
    return (feature_vector[0] ** 2) * (-np.cos(x_norm * feature_vector[0] + feature_vector[1])) * ((1 / rn) ** 2)


#sample interior collocation points
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)

# To compute the matrices
def P(x, xn, feature_vector, r):
    return (psi(x, xn, r) * second_derivative_feature(x, feature_vector,xn,r)
            + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector,xn,r)
            + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector,xn,r)
            + lam * psi(x, xn, r) * feature_function(x, feature_vector,xn,r))

# Calculate approximate solution using found coefficients
def approximate_solution(x):
    total = 0
    for n in range(len(centers)):
        pou = psi(x, centers[n], radii)
        for j in range(0, M):
            unj = U[n * M + j]
            feature_value = feature_function(x, feature_vectors_list[j],centers[n],radii)
            total += unj * feature_value * pou
    return total


for M in [20]:
    err_list = []
    for i in range(1):
        N = 3
        Q = M*N*2
        lamB = Q//20

        #calculate values
        centers, radii = centers_radii(N)

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = generate_feature_vectors(M)

        #Initialize matrices to zero
        A = np.zeros((len(centers) * M, len(centers) * M))
        B = np.zeros(len(centers) * M)

        #Choose collocation points
        collocation_points = collocation_points_interior(Q)

        #Compute matrice entries
        for N in range(len(centers)):
            for J in range(0, M):
                for n in range(len(centers)):
                    for j in range(0, M):
                        total = 0
                        for x in collocation_points:
                            P_nj = P(x, centers[n], feature_vectors_list[j], radii)
                            P_NJ = P(x, centers[N], feature_vectors_list[J], radii)
                            total += 2 * P_nj * P_NJ
                        A[N * M + J, n * M + j] += total
                        total = 0
                        for x in domain:
                            total += 2* lamB *psi(x,centers[n], radii) * feature_function(x,feature_vectors_list[j],centers[n],radii) * psi(x,centers[N], radii) * feature_function(x,feature_vectors_list[J],centers[N],radii)
                        A[N * M + J, n * M + j] += total
                total = 0
                for xi in collocation_points:
                    P_NJ = P(xi, centers[N], feature_vectors_list[J], radii)
                    total += 2 * f(xi) * P_NJ
                for xb in domain:
                    total += 2 * lamB *u_exact(xb) * psi(xb, centers[N], radii) * feature_function(xb, feature_vectors_list[J],centers[N],radii)
                B[N * M + J] = total

        #Solve to find optimal coefficients U
        U, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        points = np.linspace(domain[0], domain[1], 300)
        approximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        errors = np.abs(np.array(exact) - np.array(approximation))
        err_list.append(np.max(errors))
        plt.title("error = {:.10f}".format(np.max(errors)))
        plt.plot(points, exact, color="red")
        plt.plot(points, approximation, '--', color="blue")
        plt.show()

    print("error run " + str(M))
    print(np.mean(err_list))
    print("variance run " + str(M))
    print(np.var(err_list))


