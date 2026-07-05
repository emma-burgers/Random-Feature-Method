import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

def u_exact(x):
    return -1/2 * x * (1-x)

def f(x):
    return 1

#Returns a list of random vector '(weight, bias)' for each random feature function
#weight: Jn weights in [-R, R]^d
#bias: Jn biases in [-R, R]
def generate_feature_vectors(Jn, R):
    weights = np.random.uniform(-0.5, 0.5, size=Jn)
    biases = np.random.uniform(-1, 1, size=Jn)
    return [[weights[i], biases[i]] for i in range(Jn)]

#Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector):
    return np.tanh(x * feature_vector[0] + feature_vector[1])

#Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector):
    return feature_vector[0]*(1-np.tanh(feature_vector[0] * x  + feature_vector[1])**2)

#Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector):
    return feature_vector[0]**2*(-2*np.tanh(feature_vector[0]* x+ feature_vector[1]) * (1-np.tanh(feature_vector[0] *x + feature_vector[1])**2))

#sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(0, 1, I)

#set values

for M in [20]:
    err_list = []
    for i in range(1):
        Q = M * 5
        lamB = Q//10

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = generate_feature_vectors(M, 0)

        #Initialize matrices to zero
        A = np.zeros(( M,  M))
        B = np.zeros( M)

        #Choose collocation points
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
                        total += 2 * lamB*feature_function(xb,feature_vectors_list[j]) *feature_function(xb,feature_vectors_list[J])
                    A[J, j] = total

            total = 0
            for xi in collocation_points:
                total += 2 * f(xi) *  second_derivative_feature(xi,  feature_vectors_list[J])
            for xb in domain:
                total += 2 * lamB* u_exact(xb) * feature_function(xb, feature_vectors_list[J])
            B[J] = total

        #Solve to find optimal u_values
        U, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        #Calculate approximate solution using u_values
        def approximate_solution(x):
            total = 0
            for j in range(0,M):
                uj = U[ j]
                total += uj* feature_function(x,feature_vectors_list[j])
            return total

        points = np.linspace(domain[0], domain[1], 300)
        aproximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        errors = np.abs(np.array(exact) - np.array(aproximation))
        er = np.max(errors)
        err_list.append(er)
        plt.title("error = {:.10f}".format(er))
        plt.plot(points, exact, color="red")
        plt.plot(points, aproximation, '--', color="blue")
        plt.show()

    print("error run " + str(M))
    print(np.mean(err_list))
    print("variance run "  + str(M))
    print(np.var(err_list))
