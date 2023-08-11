# Lasso Regression Algorithms Comparison from scratch
This project aims to address the Lasso regression problem using various algorithms. Three different approaches have been implemented: ISTA (Iterative Soft-Thresholding Algorithm), ADMM (Alternating Direction Method of Multipliers), and a simulated distributed version of ADMM across multiple agents. The project provides a comparison of these algorithms based on computation times, iterations required for convergence, and convergence conditions visualized through graphs.

# Algorithms Implemented
The algorithms are implemented within the "LassoReg" class, allowing users to choose between different algorithms for training, set step-size, convergence tolerance, maximum iterations, and penalty terms.

1. Soft-Thresholding (ISTA)
In this algorithm, the Lasso regression problem is solved through iterative gradient descent until convergence. The non-differentiable L1 norm is managed using the concept of subdifferential and the soft-thresholding operator.

2. ADMM (Alternating Direction Method of Multipliers)
The Lasso problem is reformulated with an additional slack variable, and the ADMM approach is used for solving it. ADMM splits the problem into subproblems, each of which can be solved efficiently. The convergence criteria are based on primal and dual residuals.

3. Distributed ADMM
The original Lasso problem is amenable to distributed computation due to the separable structure of the observation vector and the data matrix. The problem is distributed across multiple agents, each optimizing its portion of the data while periodically exchanging information with a central node. This approach enhances scalability and efficiency for large datasets.

# Usage
To use the algorithms, instantiate the "LassoReg" class with desired parameters. The class provides flexibility in choosing algorithms and fine-tuning hyperparameters for your specific problem.

# Convergence Criteria
For ISTA, convergence is determined by the change in the solution between iterations falling below a specified threshold.

For ADMM and Distributed ADMM, convergence relies on calculating primal and dual residuals, comparing them against tolerance thresholds. When both residuals meet the convergence criteria, the algorithm terminates.

## Result and Comparisons

The algorithms were executed with the following parameters:
- Max iterations = 50000
- Step-size = 0.01
- L1-penalty = 1
- Tolerance = 1e-4
- Agents = 9 (Distributed ADMM)

The performance comparison is presented in Table 1.

| Algorithm  | R2     | Time (s) | Iterations |
|------------|--------|----------|------------|
| ISTA       | 0.7689 | 2.09     | 32417      |
| ADMM       | 0.7688 | 3.94e-4  | 3          |
| ADMM-Dist  | 0.7692 | 0.018    | 186        |

The comparison table displays the R2 scores, computation times, and iterations required to achieve convergence for each algorithm.
