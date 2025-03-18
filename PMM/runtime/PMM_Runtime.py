# hessian free V2
# runtime
import autograd.numpy as np
from autograd import hessian
from scipy.linalg import solve
import matplotlib.pyplot as plt
import math
import time
from scipy.optimize import minimize

class Functions_and_Gradients:

    # 1/2 * norm(x-e_1)^2
    def f0(self, x):
        return 0.5 * np.dot(x,x)-x[0] + 0.5

    def gradf0(self, x):
        v = np.zeros(x.size)
        v[0]=1.0
        return  x-v
    
    def create_nonsingular_matrix(self, dimension):
        H = np.zeros((dimension, dimension), dtype=int)
        for i in range(dimension):
            H[i, i] = 1+i
        return np.dot(H.T,H)

    def __init__(self, n, dimension):
        self.n = n
        self.dimension = dimension
        self.H = self.create_nonsingular_matrix(self.dimension)
        #print(self.H)
        for i in range(1,n+1):
            setattr(self, f"function_{i}", self._create_function(i,self.dimension, self.H))
            setattr(self, f"gradient_{i}", self._create_gradient(i,self.dimension, self.H))

    def _create_function(self, i, dimension, H):
        p = math.floor((i+1)/2)+1
        vector_e = [0] * dimension
        if 0 <= p - 1 < dimension:
            vector_e[p - 1] = 1
            vector_e = np.array(vector_e)
        else:
            print("ERROR: OUT OF RANGE FOR DIMENSION ", dimension)
        if(i%2==0):
            def function(x):
                x = np.array(x)
                s= np.dot(H,x)
                return 0.5*np.dot(x.T,s) + np.dot(vector_e.T,s) + 0.5*np.dot(vector_e.T,np.dot(H,vector_e))
        else:
            def function(x):
                x = np.array(x)
                s=np.dot(H,x)
                return 0.5*np.dot(x.T,s) - np.dot(vector_e.T,s) + 0.5*np.dot(vector_e.T,np.dot(H,vector_e))
        return function

    def _create_gradient(self, i, dimension, H):
        p = math.floor((i+1)/2)+1
        vector_e = [0] * dimension
        if 0 <= p - 1 < dimension:
            vector_e[p - 1] = 1
        else:
            print("ERROR: OUT OF RANGE FOR DIMENSION ", dimension)
        if(i%2==0):
            def gradient(x):
                return np.dot(H,x+vector_e)
        else:
            def gradient(x):
                return np.dot(H,x-vector_e)
        return gradient
    
class PMM:

    def jacobian_F(self, instance_Functions, n, x):
        return np.stack([getattr(instance_Functions, f"gradient_{i}")(x) for i in range(1, n + 1)])
    
    def hessian_f_beta(self, instance_Functions, n, x_star, beta):
        return hessian(lambda x: self.f_beta(instance_Functions, n, x, beta))(x_star)
    
    def f_beta(self, instance_Functions, n, x, beta):
        return sum(beta[i - 1] * getattr(instance_Functions, f"function_{i}")(x) for i in range(1, n + 1))
    
    def gradient_approximation_x_beta(self, instance_Functions, n, x, beta):
        hessian_f_beta_at_x = self.hessian_f_beta(instance_Functions, n, x, beta)
        jacobian_F_at_x = self.jacobian_F(instance_Functions, n, x)
        grad_x_star = solve(hessian_f_beta_at_x, -jacobian_F_at_x.T)
        return grad_x_star
    
    def surrogate_g(self, instance_Functions, L, n, beta_prime, x_beta, beta):
        grad_proxi_x_beta = self.gradient_approximation_x_beta(instance_Functions, n, x_beta, beta)
        return instance_Functions.f0(x_beta) + np.dot(instance_Functions.gradf0(x_beta).T, np.dot(grad_proxi_x_beta,beta_prime - beta)) + L * np.linalg.norm(beta_prime - beta)**2

    def opt_g(self, instance_Functions, L, n, beta_t, x_t):
        cons = ({'type': 'eq', 'fun': lambda beta_prime: np.sum(beta_prime) - 1})
        bounds = [(0, 1)] * len(beta_t)
        result = minimize(lambda beta_prime: self.surrogate_g(instance_Functions, L, n, beta_prime, x_t, beta_t),
                        beta_t,
                        bounds=bounds,
                        constraints=cons)
        return result.x
    
    def opt_f_beta(self, instance_Functions, n, beta_t_1, x_t):
        result = minimize(lambda x: self.f_beta(instance_Functions, n, x, beta_t_1), x0=x_t)
        return result.x

    def main(self, n, dimension, epsilon):
        L=2.0
        instance_Functions = Functions_and_Gradients(n, dimension)
        x_t = np.zeros(dimension)
        x_t[1]=1.2
        #x_t[5]=0.9
        beta_t = np.full(n, 1/n)
        start_time = time.time()
        while (np.linalg.norm(x_t)>epsilon):
            #print("Current x: ",x_t)
            beta_t_1 = self.opt_g(instance_Functions, L, n, beta_t, x_t)
            x_t_1 = self.opt_f_beta(instance_Functions, n, beta_t_1, x_t)
            beta_t = beta_t_1
            x_t = x_t_1
        end_time = time.time()
        return end_time-start_time

if __name__ == "__main__":
    feature_sizes = list(range(6, 200, 16))
    n = 4

    epsilon = 0.001
    repetitions = 4

    mean_runtimes_pmm = []
    lower_bound_pmm = []
    upper_bound_pmm = []
    pmm = PMM()

    for dimension in feature_sizes:
        print(f"Running for n = {dimension}...")
        runtimes = []
        for _ in range(repetitions):
            runtimes.append(pmm.main(n, dimension, epsilon))
        
        mean_runtime = np.mean(runtimes)
        std_deviation = np.std(runtimes)
        
        mean_runtimes_pmm.append(mean_runtime)
        lower_bound_pmm.append(mean_runtime - std_deviation)
        upper_bound_pmm.append(mean_runtime + std_deviation)

    plt.figure(figsize=(10, 8))
    plt.plot(
        feature_sizes,
        mean_runtimes_pmm,
        label='Mean Runtime (PMM)',
        color='blue',
        linewidth=2
    )

    plt.fill_between(
        feature_sizes,
        lower_bound_pmm,
        upper_bound_pmm,
        color='blue',
        alpha=0.2,
        label='1 Std Dev Range'
    )

    plt.xlabel('Feature Size (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Runtime Variability of PMM for {n} objectives and Error epsilon={epsilon}')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()