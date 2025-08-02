# optimizer.py
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from problem import DroneOptimizationProblem

def run_optimization(n_gen=10, pop_size=100, seed=1):
    problem = DroneOptimizationProblem()
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=True)
    return res
