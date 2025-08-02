import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pandas import DataFrame
from pandas.plotting import parallel_coordinates

# ---------------- Fixed Parameters ----------------
mpayload = 1.5  # kg
rho = 1.225  # kg/m^3
g = 9.81  # m/s^2
Edensity = 200  # Wh/kg
eta_motor = 0.85
eta_ESC = 0.95
kT = 0.5e-3  # empirical coefficient
trequired = 15 / 60  # in hours
VESC_max = 25  # V
Dframe_max = 0.6  # m

# Discrete Sets
nmotors_set = [4, 6, 8]
Vbattery_set = [11.1, 14.8, 22.2]
Crating_set = [25, 50, 75, 100]

# Variable bounds
n_var = 10
n_obj = 3
n_constr = 6
xl = np.array([0, 500, 10, 0.05, 0.15, 3, 1000, 0, 0, 0.3])
xu = np.array([2, 2200, 60, 0.3, 0.5, 6, 20000, 2, 3, 2.5])

# Optimization settings
n_gen = 10

# ---------------- Problem Class ----------------
class DroneOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        nmotors_idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(nmotors_set)-1)
        Vbattery_idx = np.clip(np.round(X[:, 7]).astype(int), 0, len(Vbattery_set)-1)
        Crating_idx = np.clip(np.round(X[:, 8]).astype(int), 0, len(Crating_set)-1)

        nmotors = np.array([nmotors_set[i] for i in nmotors_idx])
        KV_motor = X[:, 1]
        Imotor_max = X[:, 2]
        mmotor = X[:, 3]
        Dprop = X[:, 4]
        Pprop = X[:, 5]
        Cbattery = X[:, 6]
        Vbattery = np.array([Vbattery_set[i] for i in Vbattery_idx])
        Crating = np.array([Crating_set[i] for i in Crating_idx])
        mframe = X[:, 9]

        mbattery = (Cbattery * Vbattery) / (Edensity * 1000)
        mtotal = mpayload + mframe + mbattery + nmotors * mmotor
        Aprop = (np.pi * Dprop**2) / 4

        Tmotor_max = kT * Dprop**3 * Pprop * KV_motor * Vbattery
        Phover = (mtotal * g)**1.5 / np.sqrt(2 * rho * nmotors * Aprop)
        tflight = (Cbattery * Vbattery * eta_motor * eta_ESC) / (1000 * Phover)

        T_W = (nmotors * Tmotor_max) / (mtotal * g)
        Idraw = (nmotors * Tmotor_max) / (eta_motor * Vbattery)
        Idraw_per_motor = Tmotor_max / (eta_motor * Vbattery)

        F1 = -T_W
        F2 = -tflight
        F3 = mtotal
        F = np.column_stack([F1, F2, F3])

        g1 = 2 * mtotal * g - (nmotors * Tmotor_max)
        g2 = trequired - tflight
        g3 = Idraw - (Crating * Cbattery / 1000)
        g4 = Idraw_per_motor - Imotor_max
        g5 = Vbattery - VESC_max
        g6 = Dprop - Dframe_max
        G = np.column_stack([g1, g2, g3, g4, g5, g6])

        out["F"] = F
        out["G"] = G

# ---------------- Optimization Execution ----------------
problem = DroneOptimizationProblem()
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_gen", n_gen)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

# ---------------- Post-Processing Visualization ----------------
variable_names = [
    "Number of Motors", "KV Motor", "Motor Current Max (A)", "Motor Mass (kg)",
    "Prop Diameter (m)", "Prop Pitch (inch)", "Battery Capacity (mAh)",
    "Battery Voltage (V)", "C-rating", "Frame Mass (kg)"
]

X = res.X
F = res.F
G = res.G

nmotors_idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(nmotors_set)-1)
Vbattery_idx = np.clip(np.round(X[:, 7]).astype(int), 0, len(Vbattery_set)-1)
Crating_idx = np.clip(np.round(X[:, 8]).astype(int), 0, len(Crating_set)-1)

X[:, 0] = [nmotors_set[i] for i in nmotors_idx]
X[:, 7] = [Vbattery_set[i] for i in Vbattery_idx]
X[:, 8] = [Crating_set[i] for i in Crating_idx]

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

    # ---- Pareto Front Plot ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(-F[:, 0], -F[:, 1], F[:, 2], c='darkgray', marker='o', s=30, alpha=0.5, label='All Solutions')

    for i in range(N):
        ax.scatter(-F_best[i, 0], -F_best[i, 1], F_best[i, 2],
                   color=colors[i], s=60, marker='o', label=f'Solution {i+1}')

    ax.set_xlabel('Thrust-to-Weight Ratio (T/W)', fontsize=12)
    ax.set_ylabel('Flight Time (hours)', fontsize=12)
    ax.set_zlabel('Total Mass (kg)', fontsize=12)
    ax.set_title('Pareto Front - Final Generation', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    plt.show()

    # ---- Decision Variable Stacked Plot ----
    fig, axes = plt.subplots(n_var, 1, figsize=(15, n_var * 0.9), sharex=False)

    for i, ax in enumerate(axes):
        ticks = None
        if i == 0:
            ticks = nmotors_set
        elif i == 7:
            ticks = Vbattery_set
        elif i == 8:
            ticks = Crating_set

        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 1, max(ticks) + 1)
            ax.hlines(0, min(ticks) - 1, max(ticks) + 1, color='gray', linewidth=5, alpha=0.3)
            for j in range(N):
                x_val = X_best[j, i]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6)
        else:
            ax.hlines(0, xl[i], xu[i], color='gray', linewidth=5, alpha=0.3)
            ax.set_xlim(xl[i], xu[i])
            for j in range(N):
                x_val = X_best[j, i]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6)

        ax.set_ylabel(variable_names[i], rotation=15, labelpad=60, fontsize=9, va='center')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis='x', direction='out')

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Solution {i+1}', markerfacecolor=colors[i], markersize=6)
               for i in range(N)]
    fig.legend(handles=handles, loc='lower center', ncol=N, fontsize=9, frameon=False)

    plt.subplots_adjust(top=0.97, bottom=0.12, left=0.12, right=0.95, hspace=0.5)
    plt.show()

    # ---- Parallel Coordinates Plot ----
    df_columns = variable_names + ["-T/W", "-FlightTime", "TotalMass"]
    data = np.hstack((X, F))
    df = DataFrame(data, columns=df_columns)

    # Normalize Data per column (min-max normalization)
    df_norm = df.copy()
    for col in df_columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        if col_max - col_min != 0:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5  # Handle constant columns

    df_norm["Solution"] = "Other"
    for i in range(N):
        idx = feasible_indices[i]
        df_norm.at[idx, "Solution"] = f"Solution {i+1}"

    plt.figure(figsize=(18, 8))

    # Plot all solutions first (no legend handling here)
    parallel_coordinates(df_norm, class_column="Solution",
                        color='darkgray', alpha=0.2, linewidth=1)

    colors = plt.cm.tab10.colors[:N]
    handles = []

    for i in range(N):
        sol_label = f"Solution {i+1}"
        parallel_coordinates(df_norm[df_norm['Solution'] == sol_label],
                            class_column="Solution",
                            color=[colors[i]], alpha=0.9, linewidth=2)
        
        # Prepare custom legend handles
        handle = plt.Line2D([0], [0], color=colors[i], linewidth=2, label=sol_label)
        handles.append(handle)

    plt.ylabel("Normalized Values (min-max scaled)", fontsize=12)
    plt.title("Parallel Coordinates Plot - All Solutions with Top 5 Highlighted")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=90)

    # Add legend manually
    plt.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.show()

