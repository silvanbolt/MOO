# Pareto Navigation Gradient Descent
from scipy.optimize import minimize
from autograd import grad
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms


# start
num_iterations = 4000
#num_iterations = 2000
#for the 1 - np.exp(-2 * np.linalg.norm(x - [0.,0.])**2) function, a pareto point:
#theta = np.array([0.27568922, 0.02506266])

#for the return 0.5 * np.dot(x,x)-x[1] + 0.5 function, a pareto point:
#theta = np.array([-0.82706767, 0.02506266])
#theta = np.array([1., 0.])
theta = np.array([-1., 0.])

def F(x):
    #x0 = [0.5, 1.0]
    #return np.linalg.norm(x - x0)**2
    return 0.5 * np.dot(x,x)-x[1] + 0.5

def f1(x):
    #return 1 - np.exp(-2 * np.linalg.norm(x - [0.,0.])**2)
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) + x[0] + x[1] + 0.5

def f2(x):
    #return 1 - np.exp(-2 * np.linalg.norm(x - [1.,0.])**2)
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) - x[0] - x[1] + 0.5


#grad_F = grad(F)
#grad_f1 = grad(f1)
#grad_f2 = grad(f2)
def grad_F(x):
    return np.array([x[0], x[1] - 1])

def grad_f1(x):
    H = np.array([[1, 1], [1, 2]])
    return np.dot(H, x) + np.array([1, 1])

def grad_f2(x):
    H = np.array([[1, 1], [1, 2]])
    return np.dot(H, x) + np.array([-1, -1])

def g(theta, omega_old):
    # sum of omega = 1
    cons = ({'type': 'eq', 'fun': lambda omega: np.sum(omega) - 1})
    # each omega \geq 0
    bounds = [(0, 1)] * 2
    result = minimize(lambda omega: np.linalg.norm(scalarization(omega, theta))**2,
                      omega_old,  # initial guess for omega is 0
                      bounds=bounds,
                      constraints=cons)
    return result.x, result.fun

def scalarization(omega, theta):
    return omega[0]*grad_f1(theta) + omega[1]*grad_f2(theta)

def direction(grad_F_theta, gradients, lam):
    return grad_F_theta+ np.dot(gradients, lam)

def dual_objective(lam, grad_F_theta, gradients, phi):
    return -0.5 * np.linalg.norm(direction(grad_F_theta, gradients, lam))**2 + np.dot(lam, np.full(len(lam), phi))

def optimization(phi, gradients, grad_F_theta, lambda_old):
    bounds = [(0, None) for _ in range(gradients.shape[1])]
    result = minimize(lambda l: -dual_objective(l, grad_F_theta, gradients, phi), x0=lambda_old, bounds=bounds)  # Negative to maximize
    return result.x


#creat graphic
x_vals = np.linspace(-10, 10, 600)
y_vals = np.linspace(-10, 10, 600)
f1_vals, f2_vals = [], []
coords = []
for x in x_vals:
    for y in y_vals:
        x_vec = np.array([x, y])
        coords.append(x_vec)
        f1_vals.append(f1(x_vec))
        f2_vals.append(f2(x_vec))

function_values = pd.DataFrame(
    {
        "f1": f1_vals,
        "f2": f2_vals,
    }
)
mask = paretoset(function_values, sense=["min", "min"])
s0, s1, s2 = [], [], []
theta_values = []

#set parameters
xi = 0.1  # initial step size
alpha_init = 0.1
alpha_decay = 1 # here extrem slow decay (make it not far otherwise)
beta = 1
e = 0.01
T = num_iterations
alpha_t = alpha_init
omega_old = [0.,1.]
lambda_t = [0.,1.]
theta_values.append(theta)
s0.append(F(theta))
s1.append(f1(theta))
s2.append(f2(theta))

print("Initial theta:", theta)
print("Initial F value:", F(theta))
print("Initial f1 value:", f1(theta))
print("Initial f2 value:", f2(theta))

for iteration in range(num_iterations):
    
    #xi_t = alpha / (iteration + beta)

    # gradients at the current theta
    grad_F_theta = grad_F(theta)

    # case distinction
    omega_old, g_theta = g(theta, omega_old)
    if(g_theta<=e):
        v_t = grad_F_theta
    else:
        phi_t = alpha_t * g_theta
        gradients = np.column_stack([grad_f1(theta), grad_f2(theta)])
        lambda_t = optimization(phi_t, gradients, grad_F_theta, lambda_t)
        v_t = grad_F_theta + np.dot(gradients, lambda_t)

    #alpha_t *= alpha_decay

    theta = theta -  xi * v_t
    theta_values.append(theta)
    s0.append(F(theta))
    s1.append(f1(theta))
    s2.append(f2(theta))
    
print("Final theta:", theta)
print("Final F value:", F(theta))
print("Final f1 value:", f1(theta))
print("Final f2 value:", f2(theta))
print("Length of optimization path: ", len(s1))

function_text_F = r'$f_0(x) = \frac{1}{2} \|x - e_2\|^2$'
function_text_f1 = r'$f_1(x) = \frac{1}{2} \|A(x + e_1)\|^2$'
function_text_f2 = r'$f_2(x) = \frac{1}{2} \|A(x - e_1)\|^2$'

#plot for theta
colors = [(1 - i / (num_iterations+1), 0, i / (num_iterations+1)) for i in range((num_iterations+1))]
plt.figure()
x_coords, y_coords = zip(*theta_values)
plt.plot(x_coords, y_coords, '-', color='red', linewidth=1)
for i, (x, y) in enumerate(theta_values):
    plt.plot(x, y, 'o', color=colors[i], markersize=5)
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('x Values Transitioning from Red to Blue')

ax = plt.gca()
offset = mtransforms.ScaledTranslation(-200/72, -200/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_F, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')

plt.figure()
plt.scatter(function_values["f1"][mask], function_values["f2"][mask], c='r', label="Pareto Front")
plt.scatter(s1, s2, c='g', label="Optimization Path (T = "+str(T)+" iterations)")
#connect the points with a green line and highlight the first with black color
plt.plot(s1, s2, 'g-')
plt.scatter(s1[0], s2[0], c='b', label="Initial Starting Point on Pareto Front", edgecolors='k')


ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -65/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_F, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')


plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title(r"$\text{Front and Iteration Steps for Objective Values } f_1 \text{ and } f_2$")
plt.legend()

plt.figure()
plt.plot(s0, marker='o', color='b', linestyle='', label="F Objective Trend")
plt.xlabel("Iteration Index")
plt.ylabel("Objective Value")
plt.title(r"$f_0$"+ " Objective Value Trend (T = "+str(T)+" Iterations)")

#add function text
ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -25/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_F, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')

plt.legend()
plt.grid(True)

plt.show()

