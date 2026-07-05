import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

def u_exact(x):
    return -1 / 2 * x * (1 - x)

def f(x):
    return 1

# Returns a list of random vector '(weight, bias)' for each random feature function
# weight: Jn weights in [-R, R]^d
# bias: Jn biases in [-R, R]
def generate_feature_vectors(Jn):
    weights = np.random.uniform(-0.5, 0.5, size=Jn)
    biases = np.random.uniform(-1, 1, size=Jn)
    return [[weights[i], biases[i]] for i in range(Jn)]

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
for M in [20]:
    err = []
    for k in range(1):
        Q = 5*M
        lamB = Q // 10

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = generate_feature_vectors(M)

        # Choose collocation points
        collocation_points = collocation_points_interior(Q)

        A_feat = np.zeros((Q + 2, M))
        f_vec = np.zeros(Q + 2)

        for i, x in enumerate(collocation_points):
            for j in range(M):
                A_feat[i, j] = second_derivative_feature(x,feature_vectors_list[j])
            f_vec[i] = f(x)

        # Boundary rows
        for k, xb in enumerate(domain):
            for j in range(M):
                A_feat[Q + k, j] = np.sqrt(lamB) * feature_function(xb, feature_vectors_list[j])
            f_vec[Q + k] = np.sqrt(lamB) * u_exact(xb)

        U, _, _, _ = np.linalg.lstsq(A_feat, f_vec, rcond=None)

        points = np.linspace(domain[0], domain[1], 300)
        aproximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        errors = np.abs(np.array(exact) - np.array(aproximation))
        err.append(np.max(errors))

        plt.title("error = {:.10f}".format(np.max(errors)))
        plt.plot(points, exact, color="red")
        plt.plot(points, aproximation, '--', color="blue")
        plt.show()

    print("mean error " + str(M) )
    print(np.mean(err))
    print("var error run " + str(M))
    print(np.var(err))


