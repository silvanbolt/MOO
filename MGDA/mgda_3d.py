#epo 3d

# EPO Search

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms
import pandas as pd

x_t = np.array([0.2, 0.9, 0.5])
#x_t = np.array([-0.8, 0.6, 0.4])
beta_t = np.array([.5,.3, .2])

H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = 500

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

def mgda_step(objective_gradients):
    gradients = np.stack(objective_gradients)
    gram_matrix = gradients @ gradients.T
    weights = np.linalg.solve(gram_matrix, np.ones(len(objective_gradients)))
    weights /= np.sum(weights)
    descent_direction = np.sum(weights[:, np.newaxis] * gradients, axis=0)
    return descent_direction


#creat graphic
def create_graphics(s1, s2, s3):
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

    #ax.text(3, 2, 1.7, function_text_f1, fontsize=10, color='black')
    #ax.text(3, 2, 1.7 * 0.94, function_text_f2, fontsize=10, color='black')
    #ax.text(3, 2, 1.7 * 0.88, function_text_f3, fontsize=10, color='black')

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


s1, s2, s3 = [], [], []

s1.append(f1(x_t))
s2.append(f2(x_t))
s3.append(f3(x_t))

eta = 0.1
T = 1000

print("Initial x:", x_t)

for iteration in range(T):

    descent_dir = mgda_step([grad_f1(x_t),grad_f2(x_t)])
    x_t = x_t - eta * descent_dir

    s1.append(f1(x_t))
    s2.append(f2(x_t))
    s3.append(f3(x_t))

print("Final x:", x_t)
print("Length of optimization path: ", len(s1))

create_graphics(s1, s2, s3)