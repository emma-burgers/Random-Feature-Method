import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

# exact solution to PDE
def u_exact(x):
    return -1 / 2 * x * (1 - x)

def f(x):
    return 1


# Returns a list of random vector '(weight, bias)' for each random feature function
def generate_feature_vectors(m):
    weights = np.random.uniform(-0.5, 0.5, size=m)
    biases = np.random.uniform(-1, 1, size=m)
    return [[weights[i], biases[i]] for i in range(m)]

# Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector):
    return np.tanh(x * feature_vector[0] + feature_vector[1])

# Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector):
    return feature_vector[0] * (1 - np.tanh(feature_vector[0] * x + feature_vector[1]) ** 2)

# Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector):
    return feature_vector[0] ** 2 * (-2 * np.tanh(feature_vector[0] * x + feature_vector[1]) * (
                1 - np.tanh(feature_vector[0] * x + feature_vector[1]) ** 2))

# sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(0, 1, I)

# Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    for j in range(0, M):
        uj = U[j]
        total += uj * feature_function(x, feature_vectors_list[j])
    return total

#Set number of features M
for M in [10]:
    err = []
    for k in range(1):
        Q = 2*M
        lamB = Q // 10

        #Generate M features
        feature_vectors_list = generate_feature_vectors(M)

        #Choose collocation points
        collocation_points = collocation_points_interior(Q-2)

        #Initialize matrices
        A_feat = np.zeros((Q, M))
        f_vec = np.zeros(Q)

        # Interior equations
        for i, x in enumerate(collocation_points):
            for j in range(M):
                A_feat[i, j] = second_derivative_feature(x,feature_vectors_list[j])
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
        err.append(np.max(errors))

        # plot result
        plt.title("error = {:.10f}".format(np.max(errors)))
        plt.plot(points, exact, color="red")
        plt.plot(points, approximation, '--', color="blue")
        plt.show()

    print("mean error " + str(M) )
    print(np.mean(err))
    print("var error run " + str(M))
    print(np.var(err))


