import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

# exact solution to PDE
def u_exact(x):
    return -1 / 2 * x * (1 - x)

#forcing term of PDE
def f(x):
    return 1

# Compute N centers and the radius
def centers_radii(N):
    centers = np.linspace(domain[0], domain[1], num=N)
    return centers, (centers[1] - centers[0]) / 2

# Normalize 'x' with respect to center 'c' and radius 'r'
def normalize_coordinate(x, c, r):
    return (x - c) / r

# Evaluate PoU function for 'x' in subdomain with center 'c' and radius 'r'
def psi(x, c, r):
    x_norm = normalize_coordinate(x, c, r)

    if -5 / 4 <= x_norm < -3 / 4:
        return (1 + np.sin(2 * np.pi * x_norm)) / 2
    elif -3 / 4 <= x_norm < 3 / 4:
        return 1
    elif 3 / 4 <= x_norm < 5 / 4:
        return (1 - np.sin(2 * np.pi * x_norm)) / 2
    else:
        return 0


# Evaluate first derivative of PoU function for 'x' in subdomain with center 'c' and radius 'r'
def first_derivative_psi(x,c, r):
    x_norm = normalize_coordinate(x, c, r)

    if -5 / 4 <= x_norm < -3 / 4:
        return (np.pi * np.cos(2 * np.pi * x_norm)) * (1 / radii)
    elif -3 / 4 <= x_norm < 3 / 4:
        return 0
    elif 3 / 4 <= x_norm < 5 / 4:
        return (-np.pi * np.cos(2 * np.pi * x_norm)) * (1 / radii)
    else:
        return 0


# Evaluate second derivative of PoU function for 'x' in subdomain with center 'c' and radius 'r'
def second_derivative_psi(x, c, r):
    x_norm = normalize_coordinate(x, c, r)
    if -5 / 4 <= x_norm < -3 / 4:
        return (-2 * np.pi ** 2 * np.sin(2 * np.pi * x_norm)) * ((1 / r) ** 2)
    elif -3 / 4 <= x_norm < 3 / 4:
        return 0
    elif 3 / 4 <= x_norm < 5 / 4:
        return (2 * np.pi ** 2 * np.sin(2 * np.pi * x_norm)) * ((1 / r) ** 2)
    else:
        return 0

#Returns a list of m feature vectors '(weight, bias)'
def generate_feature_vectors(m):
    weights = np.random.uniform(-0.5, 0.5, size=m)
    biases = np.random.uniform(-1, 1, size=m)
    return [[weights[i], biases[i]] for i in range(m)]

#Evaluate feature function with parameters 'feature vector' at 'x' in subdomain with center 'c' and radius 'r'
def feature_function(x, feature_vector, c, r):
    x = normalize_coordinate(x, c, r)
    return np.tanh(x * feature_vector[0] + feature_vector[1])


#Evaluate first derivative feature function with parameters 'feature vector' at 'x' in subdomain with center 'c' and radius 'r'
def first_derivative_feature(x, feature_vector, c, r):
    x = normalize_coordinate(x, c, r)
    return feature_vector[0] * (1 - np.tanh(feature_vector[0] * x + feature_vector[1]) ** 2) * (1 / r)


#Evaluate second derivative feature function with parameters 'feature vector' at 'x' in subdomain with center 'c' and radius 'r'
def second_derivative_feature(x, feature_vector, c, r):
    x = normalize_coordinate(x, c, r)
    return feature_vector[0] ** 2 * (-2 * np.tanh(feature_vector[0] * x + feature_vector[1]) * (
            1 - np.tanh(feature_vector[0] * x + feature_vector[1]) ** 2)) * ((1 / r) ** 2)

#Sample interior collocation points
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)


# Calculate approximate solution using found coefficients
def approximate_solution(x):
    total = 0
    for n in range(len(centers)):
        pou = psi(x, centers[n], radii)
        for j in range(0, M):
            unj = U[n * M + j]
            feature_value = feature_function(x, feature_vectors_list[j], centers[n], radii)
            total += unj * feature_value * pou
    return total


# To compute the matrices (defined in notes)
def P(x, xn, feature_vector, r):
    return (psi(x, xn, r) * second_derivative_feature(x, feature_vector, xn, r)
            + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector, xn, r)
            + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector, xn, r))


# Choose number of features M, number of collocation points Q, and number of subdomains N
for M in [10]:
    error_list = []

    # Number of iterations for convergence study
    for i in range(1):
        N=2
        Q =5*N*M

        #Scaling of the boundary contribution
        lamB = Q//10

        #Find centers and radius based on N
        centers, radii = centers_radii(N)

        #Generate M features
        feature_vectors_list = generate_feature_vectors(M)

        #Choose collocation points
        collocation_points = collocation_points_interior(Q-2)

        #Compute matrice entries
        N_cols = N * M
        boundary_points = domain

        # Initializee feature matrix and forcing term matrix
        A_feat = np.zeros((Q , N*M))
        B_forc = np.zeros(Q)

        for i, x in enumerate(collocation_points):
            for n in range(N):
                for j in range(M):
                    A_feat[i, n * M + j] = P(x, centers[n], feature_vectors_list[j], radii)
            B_forc[i] = f(x)

        for k, xb in enumerate(boundary_points):
            for n in range(N):
                for j in range(M):
                    A_feat[(Q-2) + k, n * M + j] =   np.sqrt(lamB) * psi(xb, centers[n], radii) * feature_function(xb, feature_vectors_list[j], centers[n], radii)
            B_forc[(Q - 2) + k] = np.sqrt(lamB) * u_exact(xb)
        print(np.linalg.cond(A_feat))

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


    print("mean error " + str(M) )
    print(np.mean(error_list))
    print("var error run " + str(M))
    print(np.var(error_list))

