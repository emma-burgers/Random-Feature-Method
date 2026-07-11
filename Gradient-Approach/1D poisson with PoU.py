import numpy as np
import matplotlib.pyplot as plt

domain = [0,1]

# exact solution to PDE
def u_exact(x):
    return -1/2 * x * (1-x)

#forcing term of PDE
def f(x):
    return 1

# Compute N centers and the radius
def centers_radii(N):
    centers = np.linspace(domain[0], domain[1], num=N)
    return centers, (centers[1] - centers[0]) / 2

# Normalize 'x' with respect to center 'c' and radius 'r'
def normalize_coordinate(x,c,r):
    return (x-c)/r


# Evaluate PoU function for 'x' in subdomain with center 'c' and radius 'r'
def psi(x, c, r):
    x_norm = normalize_coordinate(x, c, r)

    if -5/4 <= x_norm < -3/4:
        return (1 + np.sin(2 * np.pi * x_norm)) / 2
    elif -3/4 <= x_norm < 3/4:
        return 1
    elif 3/4 <= x_norm < 5/4:
        return (1 - np.sin(2 * np.pi * x_norm)) / 2
    else:
        return 0

# Evaluate first derivative of PoU function for 'x' in subdomain with center 'c' and radius 'r'
def first_derivative_psi(x, c, r):
    x_norm = normalize_coordinate(x, c, r)
    if -5 / 4 <= x_norm < -3 / 4:
        return (np.pi * np.cos(2 * np.pi * x_norm)) * ( 1 / r)
    elif -3 / 4 <= x_norm < 3 / 4:
        return 0
    elif 3 / 4 <= x_norm < 5 / 4:
        return (-np.pi * np.cos(2 * np.pi * x_norm)) * ( 1 / r)
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

#Returns a list of m parameter vectors '(weight, bias)'
def generate_feature_vectors(m):
    weights = np.random.uniform(-0.5, 0.5 , size=m)
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

#sample interior collocation points
def collocation_points_interior(I):
    return np.random.uniform(0, 1, I)

# Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    for n in range(len(centers)):
        pou = psi(x, centers[n], radii)
        subsum = 0
        for j in range(M):
            unj = U[n * M + j]
            feature_value = feature_function(x, feature_vectors_list[j],centers[n],radii)
            subsum += unj * feature_value
        total += pou * subsum
    return total


for M in [10]:
    err_list=[]
    for i in range(1):
        #set values
        N = 2
        Q = (M * N * 5) -2
        lamB = Q//10

        #calculate values
        centers, radii = centers_radii(N)

        #For each center, we generate a list of Jn random features vectors (weight,bias)
        feature_vectors_list = generate_feature_vectors(M)

        #To compute the matrices
        def P(x, xn, feature_vector, r):
            return (psi(x, xn, r) * second_derivative_feature(x, feature_vector,xn,r)
                    + 2 * first_derivative_psi(x, xn, r) * first_derivative_feature(x, feature_vector,xn,r)
                    + second_derivative_psi(x, xn, r) * feature_function(x, feature_vector,xn,r))

        #Initialize matrices to zero
        A = np.zeros((N* M, N* M))
        B = np.zeros(N * M)

        #Choose collocation points
        collocation_points = collocation_points_interior(Q)

        #Compute matrice entries
        for N in range(len(centers)):
            for J in range(M):
                for n in range(len(centers)):
                    for j in range(M):
                        total = 0
                        for x in collocation_points:
                            P_nj = P(x, centers[n], feature_vectors_list[j], radii)
                            P_NJ = P(x, centers[N], feature_vectors_list[J], radii)
                            total += 2 * P_nj * P_NJ
                        for xb in domain:
                            total += 2 * lamB * psi(xb,centers[n], radii) * feature_function(xb,feature_vectors_list[j],centers[n],radii) * psi(xb,centers[N], radii)*feature_function(xb,feature_vectors_list[J],centers[N],radii)
                        A[N * M + J, n * M + j] = total

                total = 0
                for xi in collocation_points:
                    P_NJ = P(xi, centers[N], feature_vectors_list[J], radii)
                    total += 2 * f(xi) * P_NJ
                for xb in domain:
                    total += 2 * u_exact(xb) * lamB * psi(xb, centers[N],radii) * feature_function(xb, feature_vectors_list[J],centers[N],radii)
                B[N * M + J] = total

        #Solve to find optimal u_values
        U, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        points = np.linspace(domain[0], domain[1], 300)
        aproximation = [approximate_solution(x) for x in points]
        exact = [u_exact(x) for x in points]

        errors = np.abs(np.array(exact) - np.array(aproximation))
        max_error = np.max(errors)
        err_list.append(max_error)

        plt.title("error = {:.10f}".format(max_error))
        plt.plot(points, exact, color="red")
        plt.plot(points, aproximation, '--', color="blue")
        plt.show()

    print("error run " + str(M))
    print(np.mean(err_list))
    print("variance run " + str(M))
    print(np.var(err_list))
