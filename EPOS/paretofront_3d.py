#epo 3d

# EPO Search

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms
import pandas as pd

H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = 1000

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

def obj_beta(C,mu,a):
    if(mu!=0):
        return np.dot(C,a)
    else:
        return np.dot(C,np.ones(m))

def solve_lp_with_simplex(A, b, obj):
    """
    Solve the LP with simplex constraints:
        maximize    obj^T beta
        subject to  A^T beta >= b
                    beta >= 0
                    sum(beta) = 1
    """
    # Convert A^T beta >= b to standard form: -A^T beta <= -b
    A_ineq = -A.T
    b_ineq = -b

    # Add simplex constraint: sum(beta) = 1
    A_eq = np.ones((1, A.shape[0]))
    b_eq = np.array([1])

    # Bounds for beta
    m = A.shape[0]
    bounds = [(0, 1) for _ in range(m)]

    # Solve the linear program
    res = linprog(-obj, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed to converge: {res.message}")
    return res.x

m=3 # nr of tasks
def epo_search(theta, losses, gradients, preferences, step_size=0.01, tol=1e-6):
    s1, s2, s3, theta_values = [f1(theta)],[f2(theta)], [f3(theta)], [theta]
    for iteration in range(T):
        loss_values = np.array([loss(theta) for loss in losses])
        grad_matrix = np.column_stack([grad(theta) for grad in gradients])

        # non-uniformity with KL divergence
        weighted_norm = preferences * loss_values
        weighted_normalization = weighted_norm / np.sum(weighted_norm)
        mu = np.sum(weighted_normalization * np.log(weighted_normalization / (1 / m)))

        # adjustments (a) based on preferences and non-uniformity
        adjustments = preferences * (np.log(weighted_normalization / (1 / m)) - mu)
        C = grad_matrix.T @ grad_matrix
        J = []
        J_bar = []
        J_star = []

        for j in range(m):
            if np.dot(adjustments,C[:, j]) > 0:
                J.append(j)
            else:
                J_bar.append(j)

        max_rel_obj_values = max(weighted_norm)
        for j in range(m):
            if weighted_norm[j]==max_rel_obj_values:
                J_star.append(j)

        obj = obj_beta(C,mu,adjustments)
        A = []
        b = []
        if not J:
            for j in J_bar:
                if j in J_star:
                    continue
                A.append(C[:,j])
                b.append(0)
        else:
            for j in J_bar:
                if j in J_star:
                    continue
                A.append(C[:,j])
                b.append(np.dot(adjustments.T,C[:,j]))
        
        for j in J_star:
            A.append(C[:,j])
            b.append(0)

        beta_star = solve_lp_with_simplex(np.column_stack(A),np.array(b),obj)
        d_nd = np.dot(grad_matrix,beta_star)
        theta_new = theta - step_size * d_nd

        # Check for convergence
        if np.linalg.norm(theta_new - theta) < tol:
            return theta_new

        theta = theta_new

        # Append to list
        s1.append(f1(theta))
        s2.append(f2(theta))
        s3.append(f3(theta))
        theta_values.append(theta)

    return theta, s1, s2, s3, theta_values

#creat graphic
def create_graphics():
    x_vals = np.linspace(-10, 10, 200)#300 good
    y_vals = np.linspace(-10, 10, 200)
    z_vals = np.linspace(-10, 10, 200)
    f1_vals, f2_vals, f3_vals = [], [], []
    coords = []
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                x_vec = np.array([x, y, z])
                coords.append(x_vec)
                f1_vals.append(f1(x_vec))
                f2_vals.append(f2(x_vec))
                f3_vals.append(f3(x_vec))

    function_values = pd.DataFrame(
        {
            "f1": f1_vals,
            "f2": f2_vals,
            "f3": f3_vals
        }
    )
    mask = paretoset(function_values, sense=["min", "min", "min"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(function_values["f1"][mask], function_values["f2"][mask], function_values["f3"][mask], c='r', label="Pareto Front")



    ax.set_xlabel(r'$f_1(x)$')
    ax.set_ylabel(r'$f_2(x)$')
    ax.set_zlabel(r'$f_3(x)$')
    ax.set_title("3D Pareto Front")




    plt.legend() #or ax.legend()
    plt.show()


if __name__ == "__main__":
    
    create_graphics()