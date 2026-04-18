import numpy as np
import matplotlib.pyplot as plt

domain = [0,10]

lam = 30

def u_exact(x):
    return np.sin(2*x) + np.cos(5*x)

def f(x):
    return 26*np.sin(2*x) + 5*np.cos(5*x)

def normalize_coordinate(x,xn,r_n):
    return (x-xn)/r_n

#Returns a list of random vector '(weight, bias)' for each random feature function
#weight: Jn weights in [-R, R]^d
#bias: Jn biases in [-R, R]
def generate_feature_vectors(Jn):
    weights = np.random.uniform(-4, 10, size=Jn)
    biases = np.random.uniform(-np.pi, np.pi, size=Jn)
    return [[weights[i], biases[i]] for i in range(Jn)]

#Returns value of the feature_function using feature_vector (weight,bias) in center xn
def feature_function(x, feature_vector):
    return np.cos(x * feature_vector[0] + feature_vector[1])

#Returns the value of the first derivative of the feature_function for a point x in xn
def first_derivative_feature(x, feature_vector):
    return feature_vector[0]*(-np.sin(x * feature_vector[0] + feature_vector[1]))

#Returns the value of the second derivative of the feature_function for a point x in xn
def second_derivative_feature(x, feature_vector):
    return feature_vector[0]**2*(-np.cos(x * feature_vector[0] + feature_vector[1]))

#sample points randomly in the domain
def collocation_points_interior(I):
    return np.random.uniform(domain[0], domain[1], I)

#set values
Jn = 40

lamB = 1

#For each center, we generate a list of Jn random features vectors (weight,bias)
feature_vectors_list = generate_feature_vectors(Jn)

#To compute the matrices
def P(x, feature_vector):
    return second_derivative_feature(x, feature_vector) + lam * feature_function(x,feature_vector)

#Initialize matrices to zero
A = np.zeros((Jn,  Jn))
B = np.zeros(Jn)

#Choose collocation points
collocation_points = collocation_points_interior(300)

#Compute matrice entries

for J in range(0,Jn):
        for j in range(0, Jn):
            total = 0
            for x in collocation_points:
                P_nj = P(x, feature_vectors_list[j])
                P_NJ = P(x, feature_vectors_list[J])
                total += 2 * P_nj * P_NJ
            for xb in domain:
                total += 2 *lamB* feature_function(xb,feature_vectors_list[j]) *feature_function(xb,feature_vectors_list[J])
            A[J,  j] = total

        total = 0
        for xi in collocation_points:
            P_NJ = P(xi, feature_vectors_list[J])
            total += 2 * f(xi) * P_NJ
        for xb in domain:
            total += 2*lamB * u_exact(xb) * feature_function(xb, feature_vectors_list[J])
        B[J] = total

#Solve to find optimal u_values
U = np.linalg.solve(A, B)

#Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    subsum = 0
    for j in range(0,Jn):
        unj = U[j]
        feature_value = feature_function(x, feature_vectors_list[j])
        subsum += unj* feature_value
    total +=  subsum
    return total

x_values = np.linspace(domain[0], domain[1], 300)
aproximation = [approximate_solution(x) for x in x_values]
exact = [u_exact(x) for x in x_values]
approximation = [approximate_solution(x) for x in x_values]
exact = [u_exact(x) for x in x_values]

errors = np.abs(np.array(exact) - np.array(approximation))
max_error = np.max(errors)

plt.title(f"error = {max_error:.10f}")
#plot result
plt.plot(x_values, exact, color="red")
plt.plot(x_values, aproximation, '--', color= "blue")
plt.show()
