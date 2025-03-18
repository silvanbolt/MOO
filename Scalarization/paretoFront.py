import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) + x[0] + x[1] + 0.5

def f2(x):
    H = np.array([[1, 1], [1, 2]])
    return 0.5 * np.dot(x.T, np.dot(H, x)) - x[0] - x[1] + 0.5

def dominates(p, q):
    return np.all(p <= q) and np.any(p < q)

num_points = 1000
x_samples = np.random.uniform(-2, 2, (num_points, 2))

f1_values = np.array([f1(x) for x in x_samples])
f2_values = np.array([f2(x) for x in x_samples])
objective_values = np.column_stack((f1_values, f2_values))

pareto_front = []
for i, p in enumerate(objective_values):
    is_dominated = False
    for j, q in enumerate(objective_values):
        if i != j and dominates(q, p):
            is_dominated = True
            break
    if not is_dominated:
        pareto_front.append(p)

pareto_front = np.array(pareto_front)

plt.figure(figsize=(8, 6))
plt.scatter(f1_values, f2_values, color='gray', s=10, alpha=0.5, label='Sampled Points')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', s=20, label='Pareto Front')
plt.xlabel(r'$f_1(x)$')
plt.ylabel(r'$f_2(x)$')
plt.title('Pareto Front for '+ r'$f_1$'+ ' and '+ r'$f_2$')
plt.legend()
plt.grid(True)
plt.show()
