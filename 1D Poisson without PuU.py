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
    weights = np.random.uniform(-0.25, 0.25 , size=Jn)
    biases = np.random.uniform(-0.25, 0.25, size=Jn)
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
M = 100
Jn = 4

#For each center, we generate a list of Jn random features vectors (weight,bias)
feature_vectors_list = generate_feature_vectors(Jn, 0)

#To compute the matrices
def P(x,  feature_vector):
    return second_derivative_feature(x, feature_vector)

#Initialize matrices to zero
A = np.zeros(( Jn,  Jn))
B = np.zeros( Jn)

#Choose collocation points
collocation_points = collocation_points_interior(300)

#Compute matrice entries
for J in range(0,Jn):
    for j in range(0, Jn):
            total = 0
            for x in collocation_points:
                P_nj = P(x,  feature_vectors_list[j])
                P_NJ = P(x,  feature_vectors_list[J])
                total += 2 * P_nj * P_NJ
            for xb in domain:
                total += 2 * feature_function(xb,feature_vectors_list[j]) *feature_function(xb,feature_vectors_list[J])
            A[ J,  j] = total

    total = 0
    for xi in collocation_points:
        P_NJ = P(xi,  feature_vectors_list[J])
        total += 2 * f(xi) * P_NJ
    for xb in domain:
        total += 2 * u_exact(xb) * feature_function(xb, feature_vectors_list[J])
    B[J] = total

#Solve to find optimal u_values
U = np.linalg.solve(A, B)

#Calculate approximate solution using u_values
def approximate_solution(x):
    total = 0
    for j in range(0,Jn):
        unj = U[ j]
        total += unj* feature_function(x,feature_vectors_list[j])
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
