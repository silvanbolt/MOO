## Multi-Objective Optimization within the Pareto Set

#### Overview

This repository contains the code and experiments for the bachelor thesis **Multi‑Objective Optimization within the Pareto Set** by Silvan Bolt.  
For the full thesis text, see `thesis.pdf`

#### Abstract

Most simple machine learning methods are based on optimizing the parameters of a model with respect to one objective function, for example, by using gradient descent. But many modern machine learning applications require multi-objective optimization, which means finding an optimum with respect to several objective functions that may conflict with one another.

Finding optimal model parameters requires balancing competing objectives. To address this trade-off, we shift the algorithmic goal toward achieving Pareto-optimality, where improving one objective may come at the cost of another. The set of such Pareto-optimal points is called the Pareto set, which is often infinite. Numerous algorithms exist that take an initial point and iteratively refine it toward a Pareto optimal solution, ensuring convergence to a point in the Pareto set. This approach is known as multi-objective optimization toward the Pareto set. Since multiple Pareto-optimal solutions exist, traditional algorithms optimizing toward the Pareto set often select a solution arbitrarily.

However, in some cases, after reaching a Pareto-optimal solution, the user may wish to prioritize or refocus on a subset of the objective functions —adjusting the trade-off between them. For instance, consider the two objective functions given by the prediction risk and the unfairness score, both of which are ideally minimized. After obtaining a Pareto-optimal model, the user might decide to shift the balance between these objectives, prioritizing one over the other. In other cases, the user may wish to minimize a criterion function, constrained to solutions in the Pareto set, which inherently alters the trade-offs between the original objectives.

One approach to this problem is to restart the optimization process to find a new trade-off. However, this can lead to significant computational overhead, as it does not leverage the fact that one already has a Pareto-optimal model. Moreover, it does not necessarily guarantee convergence to the desired solution.

Motivated by this limitation, a recent line of research has introduced Optimization in the Pareto set, where an additional criterion function is optimized within the Pareto set. This function encodes the user’s preference.

An example where this is particularly relevant is in model merging, a process where the user is combining multiple pre-trained models into one model, trading off the abilities of the individual models, ideally leading to better generalization downstream.

In this thesis, we investigate various algorithms for optimization toward and within the Pareto set and compare these algorithms across different example settings. For PNGD and PMM, two algorithms that operate within the Pareto set, we analyze PNGD’s convergence behavior and PMM’s runtime as a function of feature size. Finally, we idemonstrate a practical application of optimization within the Pareto set in the context of model merging.
