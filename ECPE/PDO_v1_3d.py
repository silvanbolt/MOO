#epo 3d

# EPO Search

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.transforms as mtransforms
import pandas as pd

x_t = np.array([-1.4, 0.9, 0.5])
#x_t = np.array([-0.82706767, 0.02506266])
beta_t = np.array([.5,.3, .2])

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

def grad_f_beta(beta, x):
    return beta[0] * grad_f1(x) + beta[1] * grad_f2(x) + beta[2] * grad_f3(x)

def f_beta(beta, x):
    return beta[0] * f1(x) + beta[1] * f2(x) + beta[2] * f3(x)

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
        grad_val = np.array([np.dot(g,grad_f1(x_t_prime)),np.dot(g,grad_f2(x_t_prime)), np.dot(g,grad_f3(x_t_prime))])
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


T=10

def paretoOPT(x_t):
    beta_t = [0.3, 0.3, 0.4]
    for iteration in range(T):
        #beta_t_1 = opt_beta(x_t_prime, beta_t)
        beta_t_1 = opt_beta_gradient_descent(x_t, beta_t)

        #x_t_1 = opt_x(beta_t_1, x_t_prime)
        x_t_1 = opt_x_grad_descent(beta_t_1, x_t, eta=0.05)
        x_t = x_t_1

        beta_t = beta_t_1
    return x_t