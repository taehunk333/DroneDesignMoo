# main.py
from optimizer import run_optimization
from postprocess import plot_pareto_front, plot_decision_variables, plot_parallel_coordinates, decode_discrete_vars
from constants import *
import numpy as np

def main():
    variable_names = [
        "Number of Motors", "KV Motor", "Motor Current Max (A)", "Motor Mass (kg)",
        "Prop Diameter (m)", "Prop Pitch (inch)", "Battery Capacity (mAh)",
        "Battery Voltage (V)", "C-rating", "Frame Mass (kg)"
    ]

    res = run_optimization(n_gen=10, pop_size=100, seed=1)
    X = res.X
    F = res.F
    G = res.G

    X_decoded = decode_discrete_vars(X)
    feasible_indices = np.where(np.all(G <= 0, axis=1))[0]

    if len(feasible_indices) == 0:
        print("No feasible solutions found.")
        return

    max_solutions = 5
    X_feasible = X_decoded[feasible_indices]
    F_feasible = F[feasible_indices]

    N = min(len(X_feasible), max_solutions)
    X_best = X_feasible[:N]
    F_best = F_feasible[:N]

    plot_pareto_front(F, feasible_indices, max_solutions)
    plot_decision_variables(X_best, variable_names, max_solutions)
    plot_parallel_coordinates(X, F, G, variable_names, max_solutions)

if __name__ == "__main__":
    main()
