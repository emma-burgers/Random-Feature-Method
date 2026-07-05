import numpy as np
import matplotlib.pyplot as plt

# exact solution to PDE
def u_exact(x):
    return np.sin(2 * x) + np.cos(5 * x)


def f(x):
    return 26 * np.sin(2 * x) + 5 * np.cos(5 * x)


# We have two centers at 0 and 10, with radius 5
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
# weight: Jn weights in [-R, R]^d
# bias: Jn biases in [-R, R]
def generate_feature_vectors(Jn, R):
    weights = np.random.uniform(-8, 8, size=Jn)
    biases = np.random.uniform(-np.pi, np.pi, size=Jn)
    return [[weights[i], biases[i]] for i in range(Jn)]


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


# Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    for n in range(len(centers)):
        pou = psi(x, centers[n], radii)
        for j in range(0, M):
            unj = U[n * M + j]
            feature_value = feature_function(x, feature_vectors_list[n][j], centers[n], radii)

            total += unj * feature_value * pou
    return total


## M is the number of features
for M in [20]:
    error_list = []

    ## number of iterations for convergence study
    for i in range(1):
        domain = [0,20]
        N = 10

        Q = M * N * 2

        #Scaling of the boundary contribution
        lamB = Q // 20

        # lambda parameter in the PDE
        lam = 30


        #calculate values
        centers, radii = centers_radii(N)

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = [generate_feature_vectors(M, radii) for i in centers]

        #To compute the matrices (defined in notes)
        def P(x, xn, feature_vector, r):
            return (psi(x, xn, r) * second_derivative_feature(x, feature_vector,xn,r)
                    + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector,xn,r)
                    + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector,xn,r)
                    + lam * psi(x,xn,r) * feature_function(x,feature_vector,xn,r))

        #Choose collocation points
        collocation_points = collocation_points_interior(Q)

        #Compute matrice entries
        N_centers = len(centers)
        N_cols = N_centers * M
        boundary_points = domain  # [0, 10]

        # Feature matrix rows: Q interior + 2 boundary
        A_feat = np.zeros((Q + len(boundary_points), N_cols))
        f_vec = np.zeros(Q + len(boundary_points))


        # Interior rows — one row per collocation point
        for i, x in enumerate(collocation_points):
            for n in range(N_centers):
                for j in range(M):
                    A_feat[i, n * M + j] = P(x, centers[n], feature_vectors_list[n][j], radii)
            f_vec[i] = f(x)

        # Boundary rows — one row per boundary point
        for k, xb in enumerate(boundary_points):
            for n in range(N_centers):
                for j in range(M):
                    A_feat[Q + k, n * M + j] =  np.sqrt(lamB) *(psi(xb, centers[n], radii)
                            * feature_function(xb, feature_vectors_list[n][j], centers[n], radii)
                    )
            f_vec[Q + k] = np.sqrt(lamB) *u_exact(xb)


        #unsure yet whether to leave rcond or not.
        U, _, _, _ = np.linalg.lstsq(A_feat, f_vec, rcond=None)


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


