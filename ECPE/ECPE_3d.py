# 3d
#epo 3d

# EPO Search

import numpy as np
from scipy.optimize import minimize
from autograd import hessian
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random
import PDO_v1_3d as pdo

x0 = np.array([1.0, 0.0, 0.0])
N = 200
# number of exploration directions per point
K = 2
# step size for exploration
s1 = 0.1
s2 = 0.5

H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def f1(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) + x[0] + 0.5

def f2(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) - x[0] + 0.5

def f3(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) - x[1] + 0.5

def grad_f1(x):
    return np.dot(H, x) + np.array([1, 0, 0])

def grad_f2(x):
    return np.dot(H, x) - np.array([1, 0, 0])

def grad_f3(x):
    return np.dot(H, x) - np.array([0, 1, 0])

def pareto_optimize(x0):
    return pdo.paretoOPT(x0)

def pareto_expand(x_star):
    beta = np.random.randn(3)
    alpha = minimize_alpha(x_star)
    correction_vector = value_of_c(alpha, x_star)
    rhs = beta[0] * (grad_f1(x_star) - correction_vector) + beta[1] * (grad_f2(x_star) - correction_vector) + beta[2] * (grad_f3(x_star) -correction_vector)
    H_matrix = hessian_H(alpha, x_star)
    v = np.linalg.solve(H_matrix, rhs)
    return v, alpha

def f_alpha(x, alpha):
    return alpha[0] * f1(x) + alpha[1] * f2(x) + alpha[2] * f3(x)

def hessian_H(alpha, x_star):
    return hessian(lambda x: f_alpha(x, alpha))(x_star)

def minimize_alpha(x_star):
    alpha_initial = np.ones(3) / 3

    # constraints: alpha >= 0 and sum(alpha) = 1
    constraints = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1})
    bounds = [(0, 1)] * len(alpha_initial)
    result = minimize(lambda alpha: func(alpha, x_star), alpha_initial, bounds=bounds, constraints=constraints, method='SLSQP')
    #print("alpha0 :", result.x[0])
    #print("alpha1 :", result.x[1])
    return result.x

def func(alpha, x_star):
    combined_gradient = alpha[0] * grad_f1(x_star) + alpha[1] * grad_f2(x_star) + alpha[2] * grad_f3(x_star)
    return np.linalg.norm(combined_gradient)**2

def value_of_c(alpha, x_star):
    return alpha[0] * grad_f1(x_star) + alpha[1] * grad_f2(x_star) + alpha[2] * grad_f3(x_star)

def dominates(x_i, x_j):
    return all(x_i <= x_j) and any(x_i < x_j)


#creat graphic
def create_graphics(pareto_points):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(function_values["f1"][mask], function_values["f2"][mask], function_values["f3"][mask], c='r', label="Pareto Front")

    # generate a color gradient from red to blue
    num_points = pareto_points.shape[0]
    colors = [(1 - i / num_points, 0, i / num_points) for i in range(num_points)]  # RGB gradient from red to blue

    # plot the points with the color gradient
    #for i, point in enumerate(pareto_points):
        #ax.scatter(point[0], point[1], point[2], color=colors[i], marker='o')


    s1, s2, s3 = [], [], []
    for point in pareto_points:
        s1.append(point[0])
        s2.append(point[1])
        s3.append(point[2])


    ax.scatter(s1, s2, s3, c='r', label="Pareto front")


    function_text_f1 = r'$f_1(x) = \frac{1}{2} \|A(x + e_1)\|^2$'
    function_text_f2 = r'$f_2(x) = \frac{1}{2} \|A(x - e_1)\|^2$'
    function_text_f3 = r'$f_3(x) = \frac{1}{2} \|A(x - e_2)\|^2$'

    # 3, 2, 1.7
    #ax.text(1, 1, 0.4, function_text_f1, fontsize=10, color='black')
    #ax.text(1, 1, 0.55, function_text_f2, fontsize=10, color='black')
    #ax.text(1, 1, 0.7, function_text_f3, fontsize=10, color='black')

    ax.set_xlabel(r'$f_1(x)$')
    ax.set_ylabel(r'$f_2(x)$')
    ax.set_zlabel(r'$f_3(x)$')
    ax.set_title("3D approximated Pareto Front")

    '''
    ax2d = plt.gca()
    offset = mtransforms.ScaledTranslation(-10/72, -65/72, plt.gcf().dpi_scale_trans)
    plt.text(1.0, 1.0, preference_text, transform=ax2d.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.text(1.0, 0.93, function_text_f1, transform=ax2d.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.text(1.0, 0.86, function_text_f2, transform=ax2d.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')'''

    '''ax.text(0, 0, max(f3_vals), preference_text, fontsize=10, color='black')
    ax.text(0, 0, max(f3_vals) * 0.95, function_text_f1, fontsize=10, color='blue')
    ax.text(0, 0, max(f3_vals) * 0.90, function_text_f2, fontsize=10, color='green')'''



    plt.legend() #or ax.legend()
    plt.show()


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
            f_val = np.array([f1(x_star_i),f2(x_star_i),f3(x_star_i)])
            if not any(dominates(xj, f_val) for xj in output):
                queue.append(x_star_i)
                output.append(f_val)
            else:
                queue.append(x_star)
                print("point failed")
    
    return output  # return N Pareto stationary points


pareto_results = algorithm_1(x0, N, K, s1, s2)
print("Generated Pareto stationary points:", pareto_results)
pareto_points = np.array(pareto_results)

create_graphics(pareto_points)