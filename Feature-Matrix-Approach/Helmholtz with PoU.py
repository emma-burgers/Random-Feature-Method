import numpy as np
import matplotlib.pyplot as plt

# exact solution to PDE
def u_exact(x):
    return np.sin(2 * x) + np.cos(5 * x)


def f(x):
    return 26 * np.sin(2 * x) + 5 * np.cos(5 * x)


# lambda parameter in the PDE
lam = 30

def centers_radii(N):
    centers = np.linspace(domain[0], domain[1], num=N)
    return centers, (centers[1] - centers[0]) / 2


def normalize_coordinate(x, xn, r_n):
    return (x - xn) / r_n


# Returns POU value of x in center xn
def psi(x, xn, radii):
    x_norm = normalize_coordinate(x, xn, radii)

    if -5 / 4 <= x_norm < -3 / 4:
        return (1 + np.sin(2 * np.pi * x_norm)) / 2
    elif -3 / 4 <= x_norm < 3 / 4:
        return 1
    elif 3 / 4 <= x_norm < 5 / 4:
        return (1 - np.sin(2 * np.pi * x_norm)) / 2
    else:
        return 0


# Returns value of the first derivative of the POU in x
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


# Returns value of the second derivative of the POU in x
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


# Returns a list of random vector '(weight, bias)' for each random feature function
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


# sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)


# not being used currently
def collocation_points_boundary():
    return [domain[0], domain[1]]


# To compute the matrices (defined in notes)
def P(x, xn, feature_vector, r):
    return (psi(x, xn, r) * second_derivative_feature(x, feature_vector, xn, r)
            + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector, xn, r)
            + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector, xn, r)
            + lam * psi(x, xn, r) * feature_function(x, feature_vector, xn, r))


# Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    for n in range(len(centers)):
        pou = psi(x, centers[n], radii)
        for j in range(0, M):
            unj = U[n * M + j]
            feature_value = feature_function(x, feature_vectors_list[j], centers[n], radii)
            total += unj * feature_value * pou
    return total


## M is the number of features
for M in [30]:
    error_list = []

    ## number of iterations for convergence study
    for i in range(1):
        domain = [0,20]
        N = 10

        Q = M * N * 2

        #Scaling of the boundary contribution
        lamB = Q // 20

        #calculate values
        centers, radii = centers_radii(N)

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = generate_feature_vectors(M)

        #Choose collocation points
        collocation_points = collocation_points_interior(Q-2)

        #Initialize matrices
        A_feat = np.zeros((Q, N*M))
        B_forc = np.zeros(Q)

        #interior equations
        for i, xi in enumerate(collocation_points):
            for n in range(N):
                for j in range(M):
                    A_feat[i, n * M + j] = P(xi, centers[n], feature_vectors_list[j], radii)
            B_forc[i] = f(xi)

        # boundary equations
        for k, xb in enumerate(domain):
            for n in range(N):
                for j in range(M):
                    A_feat[(Q-2) + k, n * M + j] =  np.sqrt(lamB) *(psi(xb, centers[n], radii)
                            * feature_function(xb, feature_vectors_list[j], centers[n], radii)
                    )
            B_forc[(Q - 2) + k] = np.sqrt(lamB) * u_exact(xb)

        U, _, _, _ = np.linalg.lstsq(A_feat, B_forc, rcond=None)


        # We evaluate the found approximation on 300 points
        points = np.linspace(domain[0], domain[1], 300)
        approximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        #plot result
        errors = np.abs(np.array(exact) - np.array(approximation))
        error_list.append(np.max(errors))
        plt.title("error = {:.10f}".format(np.max(errors)))
        plt.plot(points, exact, color="red")
        plt.plot(points, approximation, '--', color="blue")
        plt.show()


    # For convergence study
    print("mean error " + str(M) )
    print(np.mean(error_list))
    print("var error run " + str(M))
    print(np.var(error_list))


