#Efficient Continuous Pareto Exploration in Multi-Task Learning

from autograd import grad
#import numpy as np
import autograd.numpy as np
from scipy.optimize import minimize
from autograd import hessian
import matplotlib.pyplot as plt
import random
import cvxpy as cp
import PDO_concave as pdo

import matplotlib.transforms as mtransforms

'''
possible starting values:
fifth pareto point:  [-0.70426065, -0.70927318]
fifth pareto point:  [-0.69924812, -0.70426065]
fifth pareto point:  [-0.69423559, -0.68922306]
fifth pareto point:  [-0.62406015, -0.62907268]
fifth pareto point:  [-0.54385965, -0.53884712]
fifth pareto point:  [-0.37343358, -0.37844612]
fifth pareto point:  [-0.20802005, -0.20802005]


'''


d=2
p = (1 / np.sqrt(d)) * np.ones(d)

def f1(x):
    return 1-np.exp(-np.linalg.norm(x-p)**2)

def f2(x):
    return 1-np.exp(-np.linalg.norm(x+p)**2)

def grad_f1(x):
    return 2*(1-f1(x))*(x-p)

def grad_f2(x):
    return 2*(1-f2(x))*(x+p)

def pareto_optimize(x0):
    return pdo.paretoOPT(x0)

def pareto_expand(x_star):
    beta = np.random.randn(2)
    alpha = minimize_alpha(x_star)
    correction_vector = value_of_c(alpha, x_star)
    rhs = beta[0] * (grad_f1(x_star) - correction_vector) + beta[1] * (grad_f2(x_star) - correction_vector)
    H_matrix = hessian_H(alpha, x_star)
    v = np.linalg.solve(H_matrix, rhs)
    return v, alpha

def f_alpha(x, alpha):
    return alpha[0] * f1(x) + alpha[1] * f2(x)

def hessian_H(alpha, x_star):
    return hessian(lambda x: f_alpha(x, alpha))(x_star)

def minimize_alpha(x_star):
    alpha_initial = np.ones(2) / 2

    # constraints: alpha >= 0 and sum(alpha) = 1
    constraints = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1})
    bounds = [(0, 1)] * len(alpha_initial)
    result = minimize(lambda alpha: func(alpha, x_star), alpha_initial, bounds=bounds, constraints=constraints, method='SLSQP')
    #print("alpha0 :", result.x[0])
    #print("alpha1 :", result.x[1])
    return result.x

def func(alpha, x_star):
    combined_gradient = alpha[0] * grad_f1(x_star) + alpha[1] * grad_f2(x_star)
    return np.linalg.norm(combined_gradient)**2

def value_of_c(alpha, x_star):
    return alpha[0] * grad_f1(x_star) + alpha[1] * grad_f2(x_star)

def dominates(x_i, x_j):
    return all(x_i <= x_j) and any(x_i < x_j)

# algo start:
def algorithm_1(x0, N, K, s1, s2):
    #x_star0 = pareto_optimize(x0)
    x_star0 = x0
    print("x_star0: ", x_star0)
    queue = [x_star0]
    output = []

    while len(output) < N:
        print("PASS! currently: ", len(output))
        x_star = queue.pop(0)
        # generate K exploration directions
        for i in range(K):
            # compute tangent direction
            v, alpha = pareto_expand(x_star)
            v = v / np.linalg.norm(v)
            s = random.uniform(s1,s2)
            # generate new point
            xi = x_star + s * v
            x_star_i = pareto_optimize(xi)
            print("passed, currently: ", len(output))
            #x_star_i = xi

            # check for dominance and add to output and queue if valid
            f_val = np.array([f1(x_star_i),f2(x_star_i)])
            if not any(dominates(xj, f_val) for xj in output):
                queue.append(x_star_i)
                output.append(f_val)
            else:
                queue.append(x_star)
                print("point failed")
    
    return output  # return N Pareto stationary points

#x0 = np.array([-0.9, 0.1])
x0 = np.array([-0.37343358, -0.37844612])
N = 20
# number of exploration directions per point
K = 10
# step size for exploration
s1 = 0.1
s2 = 0.5

pareto_results = algorithm_1(x0, N, K, s1, s2)
print("Generated Pareto stationary points:", pareto_results)
pareto_points = np.array(pareto_results)

print("Size of output: ", len(pareto_results))
plt.figure(figsize=(6, 6))

# generate a color gradient from red to blue
num_points = pareto_points.shape[0]
colors = [(1 - i / num_points, 0, i / num_points) for i in range(num_points)]  # RGB gradient from red to blue

# plot the points with the color gradient
for i, point in enumerate(pareto_points):
    plt.scatter(point[0], point[1], color=colors[i], marker='o')

plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)

function_text_f1 = r'$f_1(x) = 1 - \exp\left(-\|x - p\|^2\right)$'
function_text_f2 = r'$f_2(x) = 1 - \exp\left(-\|x + p\|^2\right)$'

ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -10/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')

plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title('Pareto Stationary Points (Red to Blue Transition)')
plt.grid(True)
plt.show()
