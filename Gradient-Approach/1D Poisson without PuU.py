import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

def u_exact(x):
    return -1/2 * x * (1-x)

#forcing term of PDE
def f(x):
    return 1

#Returns a list of m random vectors '(weight, bias)'
def generate_feature_vectors(m):
    weights = np.random.uniform(-0.5, 0.5, size=m)
    biases = np.random.uniform(-1, 1, size=m)
    return [[weights[i], biases[i]] for i in range(m)]

#Evaluate feature function with parameters 'feature vector' at 'x'
def feature_function(x, feature_vector):
    return np.tanh(x * feature_vector[0] + feature_vector[1])

#Evaluate first derivative of feature function with parameters 'feature vector' at 'x'
def first_derivative_feature(x, feature_vector):
    return feature_vector[0]*(1-np.tanh(feature_vector[0] * x  + feature_vector[1])**2)

#Evaluate second derivative of feature function with parameters 'feature vector' at 'x'
def second_derivative_feature(x, feature_vector):
    return feature_vector[0]**2*(-2*np.tanh(feature_vector[0]* x+ feature_vector[1]) * (1-np.tanh(feature_vector[0] *x + feature_vector[1])**2))

#Sample interior collocation points
def collocation_points_interior(I):
    return np.random.uniform(0, 1, I)

# Calculate approximate solution using found coefficients
def approximate_solution(x):
    total = 0
    for j in range(0, M):
        uj = U[j]
        total += uj * feature_function(x, feature_vectors_list[j])
    return total

# Choose number of features M, number of collocation points Q and penalty weight lamB for boundary
for M in [20]:
    err_list = []
    # Number of iterations i
    for i in range(1):
        Q = (M * 5) -2
        lamB = Q//10

        # Generate M parameter vectors
        feature_vectors_list = generate_feature_vectors(M)

        #Initialize matrices to zero
        A = np.zeros(( M,  M))
        B = np.zeros( M)

        # Sample collocation points
        collocation_points = collocation_points_interior(Q)

        #Compute matrice entries
        for J in range(0,M):
            for j in range(0, M):
                    total = 0
                    for x in collocation_points:
                        P_j = second_derivative_feature(x,  feature_vectors_list[j])
                        P_J = second_derivative_feature(x,  feature_vectors_list[J])
                        total += 2 * P_j * P_J
                    for xb in domain:
                        total += 2 * lamB * feature_function(xb,feature_vectors_list[j]) *feature_function(xb,feature_vectors_list[J])
                    A[J, j] = total

            total = 0
            for xi in collocation_points:
                total += 2 * f(xi) *  second_derivative_feature(xi,  feature_vectors_list[J])
            for xb in domain:
                total += 2 * lamB* u_exact(xb) * feature_function(xb, feature_vectors_list[J])
            B[J] = total

        #Solve to find optimal coefficient vector
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

    # For convergence study
    print("error run " + str(M))
    print(np.mean(err_list))
    print("variance run "  + str(M))
    print(np.var(err_list))
