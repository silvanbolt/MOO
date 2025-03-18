#Hessian free approach SML group

from autograd import grad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms

# start with (x_0,beta) on the Pareto Manifold
#x_t = np.array([1., 0.])
#beta_t = np.array([0., 1.])

#for the 1 - np.exp(-2 * np.linalg.norm(x - [0.,0.])**2) function, a pareto point:
#x_t = np.array([0.27568922, 0.02506266])
#beta_t = np.array([.5,.5])

#for the return 0.5 * np.dot(x,x)-x[1] + 0.5 function, a pareto point:
#x_t = np.array([-1.0, 0.0])

x_t = np.array([-1.4, 0.9])
#x_t = np.array([-1.6, 0.2])
#x_t = np.array([-0.82706767, 0.02506266])

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

def mgda_step(objective_gradients):
    gradients = np.stack(objective_gradients)
    gram_matrix = gradients @ gradients.T
    weights = np.linalg.solve(gram_matrix, np.ones(len(objective_gradients)))
    weights /= np.sum(weights)
    descent_direction = np.sum(weights[:, np.newaxis] * gradients, axis=0)
    return descent_direction

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

eta=0.1
T = 200

print("Initial x:", x_t)

for iteration in range(T):


    descent_dir = mgda_step([grad_f1(x_t),grad_f2(x_t)])
    x_t = x_t - eta * descent_dir

    
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