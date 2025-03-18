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
def create_graphics(preferences, s1, s2, s3, theta_values):
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

    ax.plot(s1, s2, s3, 'g-', label="Optimization Path")
    ax.scatter(s1[0], s2[0], s3[0], c='b', label="Initial Starting Point", edgecolors='k')

    function_text_f1 = r'$f_1(x) = \frac{1}{2} \|A(x + e_1)\|^2$'
    function_text_f2 = r'$f_2(x) = \frac{1}{2} \|A(x - e_1)\|^2$'
    function_text_f3 = r'$f_3(x) = \frac{1}{2} \|A(x - e_2)\|^2$'
    preference_text = rf'$\text{{Preferences: }}({preferences[0]:.1f},{preferences[1]:.1f},{preferences[2]:.1f})$'

    #ax.text(3, 2, 1.7, function_text_f1, fontsize=10, color='black')
    #ax.text(3, 2, 1.7 * 0.94, function_text_f2, fontsize=10, color='black')
    #ax.text(3, 2, 1.7 * 0.88, function_text_f3, fontsize=10, color='black')
    #ax.text(3, 2, 1.7 * 0.82, preference_text, fontsize=10, color='black')

    ax.set_xlabel(r'$f_1(x)$')
    ax.set_ylabel(r'$f_2(x)$')
    ax.set_zlabel(r'$f_3(x)$')
    ax.set_title("3D Pareto Front and Optimization Path")

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




    '''plt.figure()
    plt.scatter(function_values["f1"][mask], function_values["f2"][mask], c='r', label="Pareto Front")
    
    plt.scatter(s1, s2, c='g', label="Optimization Path (T = "+str(T)+" iterations)")
    #connect the points with a green line and highlight the first with black color
    plt.plot(s1, s2, 'g-')
    plt.scatter(s1[0], s2[0], c='b', label="Initial Starting Point on Pareto Front", edgecolors='k')

    function_text_f1 = r'$f_1(x) = \frac{1}{2} \|A(x + e_1)\|^2$'
    function_text_f2 = r'$f_2(x) = \frac{1}{2} \|A(x - e_1)\|^2$'
    preference_text = rf'$\text{{Preferences: }} = ({preferences[0]:.1f},{preferences[1]:.1f})$'

    ax = plt.gca()
    offset = mtransforms.ScaledTranslation(-10/72, -65/72, plt.gcf().dpi_scale_trans)
    plt.text(1.0, 1.0, preference_text, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
    

    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title("Front and Iteration Steps for Objective Values f1 and f2")
    plt.legend()
    plt.show()'''

if __name__ == "__main__":
    #theta_0 = np.array([0.2, 0.6])
    #theta_0 = np.array([-0.8, 1.3, 1.2])
    #theta_0 = np.array([-0.82706767, 0.02506266])


    #theta_0 = np.array([-0.8, 0.6, 0.4])
    theta_0 = np.array([0.2, 0.9, 0.5])


    #equal preference for f1 and f2:
    #!!!!!
    #preferences = np.array([0.33, .33, .33])#only one nachkommastelle in graph!!!
    preferences = np.array([0.1, .1, .8])
    #!!!!!!!!
    step_size = 0.01

    # Run EPO Search
    theta_opt, s1, s2, s3, theta_values = epo_search(
        theta=theta_0,
        losses=[f1, f2, f3],
        gradients=[grad_f1, grad_f2, grad_f3],
        preferences=preferences,
        step_size=step_size
    )

    print("Optimal solution:", theta_opt)
    print("Objective 1:", f1(theta_opt))
    print("Objective 2:", f2(theta_opt))
    print("Objective 3:", f3(theta_opt))

    create_graphics(preferences, s1, s2, s3, theta_values)