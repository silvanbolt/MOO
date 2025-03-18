#scalarization for different weights

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f1(x):
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) + x[0] + x[1] + 0.5

def f2(x):
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) - x[0] - x[1] + 0.5

def pareto_search(x0, num_points=50, step_size=0.05):
    pareto_points = []

    for i in range(num_points):
        w1 = i / (num_points - 1)
        weights = [w1, 1 - w1]

        def combined_objective(x):
            return weights[0] * f1(x) + weights[1] * f2(x)

        result = minimize(combined_objective, x0, method='SLSQP', options={'ftol': 1e-6})
        if result.success:
            pareto_points.append([f1(result.x), f2(result.x)])
            x0 = result.x + step_size * np.random.randn(len(x0))

    return np.array(pareto_points)

x0 = np.array([0.0, 0.0])

pareto_points = pareto_search(x0)

plt.figure(figsize=(8, 6))
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', label='Pareto Points')
plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title('Pareto Front Approximation Using Pareto Search Algorithm')
plt.legend()
plt.grid(True)
plt.show()
