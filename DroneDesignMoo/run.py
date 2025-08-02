import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pandas import DataFrame
from pandas.plotting import parallel_coordinates

# ---------------- Fixed Parameters ----------------
mpayload = 1.5  # kg
rho = 1.225  # kg/m^3
g = 9.81  # m/s^2
Edensity = 200  # Wh/kg
eta_motor = 0.85
eta_ESC = 0.95
kT = 0.5e-3  # empirical coefficient (adjust as needed)
trequired = 15 / 60  # in hours
VESC_max = 25  # V
Dframe_max = 0.6  # m

# Discrete Sets
nmotors_set = [4, 6, 8]
Vbattery_set = [11.1, 14.8, 22.2]
Crating_set = [25, 50, 75, 100]

# ---------------- Variable Encoding ----------------
n_var = 11
n_obj = 3
n_constr = 6

# Variable bounds
xl = np.array([0, 500, 10, 0.05, 0.15, 3, 1000, 0, 0, 0.3, 3])
xu = np.array([2, 2200, 60, 0.3, 0.5, 6, 20000, 2, 3, 2.5, 6])

class DroneOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

problem = DroneOptimizationProblem()

algorithm = NSGA2(pop_size=100)
termination = NoTermination()
algorithm.setup(problem, termination=termination)

np.random.seed(1)

# ---------------- Real-Time Plot Setup ----------------
plt.ion()
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# ---------------- Optimization Loop ----------------
for n_gen in range(10):
    pop = algorithm.ask()
    X = pop.get("X")

    # Manually Round Discrete Variables
    nmotors_idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(nmotors_set)-1)
    Vbattery_idx = np.clip(np.round(X[:, 7]).astype(int), 0, len(Vbattery_set)-1)
    Crating_idx = np.clip(np.round(X[:, 8]).astype(int), 0, len(Crating_set)-1)

    # Decode Variables
    nmotors = np.array([nmotors_set[i] for i in nmotors_idx])
    KV_motor = X[:, 1]
    Imotor_max = X[:, 2]
    mmotor = X[:, 3]
    Dprop = X[:, 4]
    Pprop = X[:, 10]
    Cbattery = X[:, 6]
    Vbattery = np.array([Vbattery_set[i] for i in Vbattery_idx])
    Crating = np.array([Crating_set[i] for i in Crating_idx])
    mframe = X[:, 9]

    # Derived Quantities
    mbattery = (Cbattery * Vbattery) / (Edensity * 1000)
    mtotal = mpayload + mframe + mbattery + nmotors * mmotor
    Aprop = (np.pi * Dprop**2) / 4

    # Motor Maximum Thrust
    Tmotor_max = kT * Dprop**3 * Pprop * KV_motor * Vbattery

    # Hover Power
    Phover = (mtotal * g)**1.5 / np.sqrt(2 * rho * nmotors * Aprop)

    # Flight Time (hours)
    tflight = (Cbattery * Vbattery * eta_motor * eta_ESC) / (1000 * Phover)

    # Thrust-to-Weight Ratio
    T_W = (nmotors * Tmotor_max) / (mtotal * g)

    # Battery Current Draw
    Idraw = (nmotors * Tmotor_max) / (eta_motor * Vbattery)
    Idraw_per_motor = Tmotor_max / (eta_motor * Vbattery)

    # Objectives
    F1 = -T_W
    F2 = -tflight
    F3 = mtotal
    F = np.column_stack([F1, F2, F3])

    # Constraints
    g1 = 2 * mtotal * g - (nmotors * Tmotor_max)
    g2 = trequired - tflight
    g3 = Idraw - (Crating * Cbattery / 1000)
    g4 = Idraw_per_motor - Imotor_max
    g5 = Vbattery - VESC_max
    g6 = Dprop - Dframe_max
    G = np.column_stack([g1, g2, g3, g4, g5, g6])

    pop.set("F", F)
    pop.set("G", G)

    algorithm.tell(infills=pop)

    # ---- Real-Time Plot Update ----
    ax.clear()
    ax.scatter(-F[:, 0], -F[:, 1], F[:, 2], c='red', marker='o')
    ax.set_xlabel('Thrust-to-Weight Ratio (T/W)', fontsize=12)
    ax.set_ylabel('Flight Time (hours)', fontsize=12)
    ax.set_zlabel('Total Mass (kg)', fontsize=12)
    ax.set_title(f'Pareto Front - Generation {algorithm.n_gen}', fontsize=14)
    plt.pause(0.5)

    print(f"Generation {algorithm.n_gen}")

plt.ioff()
plt.show()

# ---------------- Final Decision Variable Plot ----------------
variable_names = [
    "Number of Motors", "KV Motor", "Motor Current Max (A)", "Motor Mass (kg)",
    "Prop Diameter (m)", "Prop Pitch (inch)", "Battery Capacity (mAh)",
    "Battery Voltage (V)", "C-rating", "Frame Mass (kg)", "Prop Pitch (cm)"
]

# Get final population and their decoded variables
res = algorithm.pop
X = res.get("X")
F = res.get("F")
G = res.get("G")

# Filter feasible solutions
feasible_indices = np.where(np.all(G <= 0, axis=1))[0]
X_feasible = X[feasible_indices]
F_feasible = F[feasible_indices]

if len(X_feasible) == 0:
    print("No feasible solutions found.")
else:
    max_solutions = 5
    N = min(len(X_feasible), max_solutions)
    X_best = X_feasible[:N]
    F_best = F_feasible[:N]

    colors = plt.cm.tab10.colors[:N]

    # ---- 3D Pareto plot: All solutions in gray, first 5 feasible colored ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all solutions in light gray
    ax.scatter(-F[:, 0], -F[:, 1], F[:, 2], c='lightgray', marker='o', s=30, alpha=0.5, label='All Solutions')

    # Highlight first 5 feasible with colors
    for i in range(N):
        ax.scatter(-F_best[i, 0], -F_best[i, 1], F_best[i, 2], 
                   color=colors[i], s=60, marker='o', label=f'Solution {i+1}')

    ax.set_xlabel('Thrust-to-Weight Ratio (T/W)', fontsize=12)
    ax.set_ylabel('Flight Time (hours)', fontsize=12)
    ax.set_zlabel('Total Mass (kg)', fontsize=12)
    ax.set_title('Pareto Front - Final Generation', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    plt.show()

    # ---- Compact stacked subplots for decision variables (only 5 colored solutions) ----
    fig, axes = plt.subplots(n_var, 1, figsize=(15, n_var * 0.9), sharex=False)

    for i, ax in enumerate(axes):
        if i == 0:
            # Number of Motors: plot actual discrete values on x-axis
            ticks = nmotors_set
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 1, max(ticks) + 1)
            ax.hlines(0, min(ticks) - 1, max(ticks) + 1, color='gray', linewidth=5, alpha=0.3)
            for j in range(N):
                nmotors_idx = int(np.clip(np.round(X_best[j, i]), 0, len(nmotors_set)-1))
                x_val = nmotors_set[nmotors_idx]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6,
                        label=f'Solution {j+1}' if i == 0 else "")
        elif i == 7:
            # Battery Voltage: plot actual discrete values
            ticks = Vbattery_set
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 1, max(ticks) + 1)
            ax.hlines(0, min(ticks) - 1, max(ticks) + 1, color='gray', linewidth=5, alpha=0.3)
            for j in range(N):
                Vbattery_idx = int(np.clip(np.round(X_best[j, i]), 0, len(Vbattery_set)-1))
                x_val = Vbattery_set[Vbattery_idx]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6,
                        label=f'Solution {j+1}' if i == 7 else "")
        elif i == 8:
            # C-rating: plot actual discrete values
            ticks = Crating_set
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 5, max(ticks) + 5)
            ax.hlines(0, min(ticks) - 5, max(ticks) + 5, color='gray', linewidth=5, alpha=0.3)
            for j in range(N):
                Crating_idx = int(np.clip(np.round(X_best[j, i]), 0, len(Crating_set)-1))
                x_val = Crating_set[Crating_idx]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6,
                        label=f'Solution {j+1}' if i == 8 else "")
        else:
            ax.hlines(0, xl[i], xu[i], color='gray', linewidth=5, alpha=0.3)
            ax.set_xlim(xl[i], xu[i])
            for j in range(N):
                x_val = X_best[j, i]
                x_val_clipped = np.clip(x_val, xl[i], xu[i])
                ax.plot(x_val_clipped, 0, 'o', color=colors[j], markersize=6,
                        label=f'Solution {j+1}' if i == 0 else "")

        ax.set_ylabel(variable_names[i], rotation=15, labelpad=60, fontsize=9, va='center')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis='x', direction='out')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=N, fontsize=9, frameon=False)

    plt.subplots_adjust(top=0.97, bottom=0.12, left=0.12, right=0.95, hspace=0.5)
    plt.show()

    # ---- Parallel Coordinates Plot ----
    df_columns = variable_names + ["-T/W", "-FlightTime", "TotalMass"]
    data = np.hstack((X_best, F_best))
    df = DataFrame(data, columns=df_columns)

    plt.figure(figsize=(18, 8))
    parallel_coordinates(df, class_column=None, colormap=plt.get_cmap("viridis"))
    plt.title("Parallel Coordinates Plot - Decision Variables and Objectives")
    plt.grid(True, alpha=0.3)
    plt.show()
