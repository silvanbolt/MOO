#PMM

#import numpy as np
from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from autograd import hessian
from scipy.optimize import minimize
from scipy.linalg import solve
import matplotlib.transforms as mtransforms

T=50

#for the 1 - np.exp(-2 * np.linalg.norm(x - [0.,0.])**2) function, a pareto point:
#x_init = np.array([0.27568922, 0.02506266])
#beta_init = np.array([1/2, 1/2])

#for the return 0.5 * np.dot(x,x)-x[1] + 0.5 function, a pareto point:
#x_t = np.array([-0.82706767, 0.02506266])
#x_t = np.array([1.0, 0.0])
x_t = np.array([0.192, -1.292])
beta_t = np.array([0.5, 0.5])

A_french = np.array([[1.2471149342574281, -0.005815372666163833], [-0.005815372666163833, 0.03213382494519598]])#row-wise
b_french = np.array([-0.4957788839441924,0.0853532924993083])
e_french = 13.938305924444245

A_arabic = np.array([[1.2404395671057493,0.1537708665830773], [0.1537708665830773,0.10887202130381757]])
b_arabic = np.array([-2.1526517935731357,-0.4858033016931851])
e_arabic = 21.522488344282003

A_japanese = np.array([[6.669501781986549,-0.10456911026312053], [-0.10456911026312053,0.3025668758954015]])
b_japanese = np.array([-4.591681448958229,-0.05051354219084881])
e_japanese = 47.6160027554793

def f0(x):
    return e_japanese + np.dot(b_japanese, x) +  0.5 * np.dot(x.T, np.dot(A_japanese, x))

def f1(x):
    return e_french + np.dot(b_french, x) +  0.5 * np.dot(x.T, np.dot(A_french, x))

def f2(x):
    return e_arabic + np.dot(b_arabic, x) +  0.5 * np.dot(x.T, np.dot(A_arabic, x))

# grid points for x in the range [-10, 10]
x_vals = np.linspace(-10, 10, 800)
y_vals = np.linspace(-10, 10, 800)

# objective functions values for each point in the grid
f1_vals, f2_vals = [], []
coords = []

for x in x_vals:
    for y in y_vals:
        x_vec = np.array([x, y])
        coords.append(x_vec)
        f1_vals.append(f1(x_vec))
        f2_vals.append(f2(x_vec))

# computing the gradient
grad_f0 = grad(f0)
grad_f1 = grad(f1)
grad_f2 = grad(f2)

# initial objective values
print("initial f0(x):", f0(x_t))
print("initial f1(x):", f1(x_t))
print("initial f2(x):", f2(x_t))
print("initial x: ", x_t)

# run the PMM algorithm
s0, s1, s2 = [], [], []
x_values = []

s0.append(f0(x_t))
s1.append(f1(x_t))
s2.append(f2(x_t))
x_values.append(x_t)

# jacobian of F
def jacobian_F(x):
    return np.stack([grad_f1(x), grad_f2(x)])

# hessian of f_beta
def hessian_f_beta(x_star, beta):
    return hessian(lambda x: f_beta(x, beta))(x_star)

# f_beta denotes the scalarization with convex weights beta
def f_beta(x, beta):
    return beta[0] * f1(x) + beta[1] * f2(x)

def gradient_approximation_x_beta(x, beta):
    # compute the hessian of f_beta at x
    hessian_f_beta_at_x = hessian_f_beta(x, beta)
    # compute the jacobian of the individual objective functions at x
    jacobian_F_at_x = jacobian_F(x)
    # solve hessian_f_beta_at_x_beta * v = -jacobian_F_at_x.T for v,
    grad_x_star = solve(hessian_f_beta_at_x, -jacobian_F_at_x.T)
    return grad_x_star

# wooooowwww, adjust the value of the lipschitz constant: (in paper is 0.5)
L=4
def surrogate_g(beta_prime, x_beta, beta):
    grad_proxi_x_beta = gradient_approximation_x_beta(x_beta, beta)
    return f0(x_beta) + np.dot(grad_f0(x_beta).T, np.dot(grad_proxi_x_beta,beta_prime - beta)) + L * np.linalg.norm(beta_prime - beta)**2

def opt_g(beta_t, x_t):
    # sum of beta_i = 1
    cons = ({'type': 'eq', 'fun': lambda beta_prime: np.sum(beta_prime) - 1})
    # each beta_i \geq 0
    bounds = [(0, 1)] * len(beta_t)
    # minimize the surrogate function g with respect to beta_prime
    result = minimize(lambda beta_prime: surrogate_g(beta_prime, x_t, beta_t),
                      beta_t,  #initial guess for beta_prime is beta itself
                      bounds=bounds,
                      constraints=cons)
    return result.x

def opt_f_beta(beta_t_1, x_t):
    result = minimize(lambda x: f_beta(x, beta_t_1), x0=x_t)
    return result.x


for iteration in range(T):
    beta_t_1 = opt_g(beta_t, x_t)
    x_t_1 = opt_f_beta(beta_t_1, x_t)
    beta_t = beta_t_1
    x_t = x_t_1
    s0.append(f0(x_t))
    s1.append(f1(x_t))
    s2.append(f2(x_t))
    x_values.append(x_t)

# final objective values
print("final m0(x): ", f0(x_t))
print("final m1(x): ", f1(x_t))
print("final m2(x): ", f2(x_t))
print("final c: ", x_t)

function_text_f0 = r'$f_0(x) = e_j + b_j^T c + \frac{1}{2} c^T A_j c $'
function_text_f1 = r'$f_1(x) = e_f + b_f^T c + \frac{1}{2} c^T A_f c $'
function_text_f2 = r'$f_2(x) = e_a + b_a^T c + \frac{1}{2} c^T A_a c $'

#plot for theta
colors = [(1 - i / (T+1), 0, i / (T+1)) for i in range((T+1))]
plt.figure()
for i, (x, y) in enumerate(x_values):
    plt.plot(x, y, 'o', color=colors[i], markersize=5)
plt.xlabel('c[0]')
plt.ylabel('c[1]')
plt.title('c values transitioning from red to blue')

ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -25/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_f0, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')

function_values = pd.DataFrame(
    {
        "f1": f1_vals,
        "f2": f2_vals,
    }
)
mask = paretoset(function_values, sense=["min", "min"])
print("Length of optimization path: ", len(s1))

plt.figure()
plt.scatter(function_values["f1"][mask], function_values["f2"][mask], c='r', label="Pareto Front")
plt.scatter(s1, s2, c='g', label="Optimization Path (T = "+str(T)+" iterations)")
#connect the points with a green line and highlight the first with black color
plt.plot(s1, s2, 'g-')
plt.scatter(s1[0], s2[0], c='b', label="Initial Starting Point on Pareto Front", edgecolors='k')

ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -65/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_f0, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')


plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title("Front and Iteration Steps for Objective Values f1 and f2")
plt.legend()

plt.figure()
plt.plot(s0, marker='o', color='b', linestyle='', label="F Objective Trend")
plt.xlabel("Iteration Index")
plt.ylabel("Objective Value")
plt.title(r"$f_0$" + " Objective Value Trend (T = "+str(T)+" Iterations)")

#add function text
ax = plt.gca()
offset = mtransforms.ScaledTranslation(-10/72, -25/72, plt.gcf().dpi_scale_trans)
plt.text(1.0, 1.0, function_text_f0, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.93, function_text_f1, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(1.0, 0.86, function_text_f2, transform=ax.transAxes + offset, fontsize=10, verticalalignment='top', horizontalalignment='right')

plt.legend()
plt.grid(True)

plt.show()