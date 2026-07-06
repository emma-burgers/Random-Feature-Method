import numpy as np
import matplotlib.pyplot as plt

# exact solution to PDE
def u_exact(x):
    return np.sin(2 * x) + np.cos(5 * x)

def f(x):
    return 26 * np.sin(2 * x) + 5 * np.cos(5 * x)

def normalize_coordinate(x, xn, r_n):
    return (x - xn) / r_n

# Returns a list of random vector '(weight, bias)' for each random feature function
def generate_feature_vectors(m):
    weights = np.random.uniform(-8, 8, size=m)
    biases = np.random.uniform(-np.pi, np.pi, size=m)
    return [[weights[i], biases[i]] for i in range(m)]


# Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector):
    return np.cos(x * feature_vector[0] + feature_vector[1])


# Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector):
    return feature_vector[0] * (-np.sin(x * feature_vector[0] + feature_vector[1]))


# Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector):
    return feature_vector[0] ** 2 * (-np.cos(x * feature_vector[0] + feature_vector[1]))


# sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)

# To compute the matrices
def P(x, feature_vector):
    return second_derivative_feature(x, feature_vector) + lam * feature_function(x, feature_vector)

# Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    subsum = 0
    for j in range(0, M):
        unj = U[j]
        feature_value = feature_function(x, feature_vectors_list[j])
        subsum += unj * feature_value
    total += subsum
    return total


for M in [80]:
    err_list = []
    for i in range(1):
        domain = [0,20]

        lam = 30

        Q = M*5
        lamB = Q // 20

        # Generate M features
        feature_vectors_list = generate_feature_vectors(M)


        #Choose collocation points
        collocation_points = collocation_points_interior(Q-2)

        #Initialize matrices
        A_feat = np.zeros((Q, M))
        f_vec = np.zeros(Q)

        #Interior equations
        for i, x in enumerate(collocation_points):
            for j in range(M):
                A_feat[i, j] = P(x, feature_vectors_list[j])
            f_vec[i] = f(x)

        # Boundary equations
        for k, xb in enumerate(domain):
            for j in range(M):
                A_feat[(Q-2) + k, j] = np.sqrt(lamB) * feature_function(xb, feature_vectors_list[j])
            f_vec[(Q-2) + k] = np.sqrt(lamB) * u_exact(xb)

        U, _, _, _ = np.linalg.lstsq(A_feat, f_vec, rcond=None)

        points = np.linspace(domain[0], domain[1], 300)
        approximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        errors = np.abs(np.array(exact) - np.array(approximation))
        max_error = np.max(errors)
        err_list.append(max_error)

        #plot result
        plt.title("error = {:.10f}".format(np.max(errors)))
        plt.plot(points, exact, color="red")
        plt.plot(points, approximation, '--', color="blue")
        plt.show()

    print("mean error " + str(M) )
    print(np.mean(err_list))
    print("var error run " + str(M))
    print(np.var(err_list))

