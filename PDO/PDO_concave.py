#PDO concave
#Hessian free approach SML group

from autograd import grad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms


#x_t = np.array([-1.0, 0.9])


#x_t = np.array([-1.4, 0.9])
x_t = np.array([-1.6, 0.2])


beta_t = np.array([.5,.5])
T = 200

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

def grad_f_beta(beta, x):
    return beta[0] * grad_f1(x) + beta[1] * grad_f2(x)

def f_beta(beta, x):
    return beta[0] * f1(x) + beta[1] * f2(x)

# beta_i sum up to 1 and bounded between 0 and 1
def project_onto_simplex(beta):
    beta = np.maximum(beta, 0)
    sorted_beta = np.sort(beta)[::-1]
    cumulative_sum = np.cumsum(sorted_beta) - 1
    rho = np.where(sorted_beta - cumulative_sum / (np.arange(len(beta)) + 1) > 0)[0][-1]
    theta = cumulative_sum[rho] / (rho + 1)
    return np.maximum(beta - theta, 0)

def opt_beta_gradient_descent(x_t_prime, beta_t, learning_rate=0.01, max_iter=1000, tol=1e-6):
    beta = beta_t
    for i in range(max_iter):
        g = 2 * grad_f_beta(beta, x_t_prime)
        grad_val = np.array([np.dot(g,grad_f1(x_t_prime)),np.dot(g,grad_f2(x_t_prime))])
        beta = beta - learning_rate * grad_val
        beta = project_onto_simplex(beta)
        if np.linalg.norm(grad_val) < tol:
            break
    return beta
    
def opt_x_grad_descent(beta_t_1, x_t_prime, eta):
    return gradient_descent_opt_x(
        lambda x: f_beta(beta_t_1, x),
        lambda x: grad_f_beta(beta_t_1, x),
        x_t_prime,
        eta=eta
    )

def gradient_descent_opt_x(func, grad_func, x0, eta, tol=1e-6, max_iter=1000):
    x = x0
    grad = grad_func(x)
    x_new = x - eta * grad
    return x_new

#creat graphic
x_vals = np.linspace(-10, 10, 800)
y_vals = np.linspace(-10, 10, 800)
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
s1, s2 = [], []

s1.append(f1(x_t))
s2.append(f2(x_t))

print("Initial x:", x_t)

for iteration in range(T):

    #beta_t_1 = opt_beta(x_t_prime, beta_t)
    beta_t_1 = opt_beta_gradient_descent(x_t, beta_t)

    #x_t_1 = opt_x(beta_t_1, x_t_prime)
    #x_t_1 = opt_x_grad_descent(beta_t_1, x_t, eta=0.2)
    x_t_1 = x_t - beta_t_1[0]*grad_f1(x_t) - beta_t_1[1]*grad_f2(x_t)
    x_t = x_t_1

    beta_t = beta_t_1
    s1.append(f1(x_t))
    s2.append(f2(x_t))

print("Final x:", x_t)
print("Length of optimization path: ", len(s1))

plt.figure()
plt.scatter(function_values["f1"][mask], function_values["f2"][mask], c='r', label="Pareto Front")
plt.scatter(s1, s2, c='g', label="Optimization Path (T = "+str(T)+" iterations)")
#connect the points with a green line and highlight the first with black color
plt.plot(s1, s2, 'g-')
plt.scatter(s1[0], s2[0], c='b', label="Initial Starting Point on Pareto Front", edgecolors='k')

function_text_f1 = r'$f_1(x) = 1 - \exp\left(-\|x - p\|^2\right)$'
function_text_f2 = r'$f_2(x) = 1 - \exp\left(-\|x + p\|^2\right)$'

ax = plt.gca()
offset = mtransforms.ScaledTranslation(-200/72, -100/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')


plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title("Front and Iteration Steps for Objective Values f1 and f2")
plt.legend()

plt.legend()
plt.grid(True)

plt.show()
