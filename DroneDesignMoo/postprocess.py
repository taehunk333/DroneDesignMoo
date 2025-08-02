# postprocess.py
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import parallel_coordinates
from DroneDesignMoo.constants import *
import matplotlib

def decode_discrete_vars(X):
    nmotors_idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(NMOTORS_SET) - 1)
    Vbattery_idx = np.clip(np.round(X[:, 7]).astype(int), 0, len(VBATTERY_SET) - 1)
    Crating_idx = np.clip(np.round(X[:, 8]).astype(int), 0, len(CRATING_SET) - 1)

    X_decoded = X.copy()
    X_decoded[:, 0] = [NMOTORS_SET[i] for i in nmotors_idx]
    X_decoded[:, 7] = [VBATTERY_SET[i] for i in Vbattery_idx]
    X_decoded[:, 8] = [CRATING_SET[i] for i in Crating_idx]
    return X_decoded

def plot_pareto_front(F, feasible_indices, max_solutions=5):
    feasible_F = F[feasible_indices]
    N = min(len(feasible_indices), max_solutions)
    colors = matplotlib.cm.tab10.colors[:N]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(-F[:, 0], -F[:, 1], F[:, 2], c='darkgray', marker='o', s=30, alpha=0.5, label='All Solutions')

    for i in range(N):
        ax.scatter(-feasible_F[i, 0], -feasible_F[i, 1], feasible_F[i, 2],
                   color=colors[i], s=60, marker='o', label=f'Solution {i+1}')

    ax.set_xlabel('Thrust-to-Weight Ratio (T/W)', fontsize=12)
    ax.set_ylabel('Flight Time (hours)', fontsize=12)
    ax.set_zlabel('Total Mass (kg)', fontsize=12)
    ax.set_title('Pareto Front - Final Generation', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    plt.show()

def plot_decision_variables(X_best, variable_names, max_solutions=5):
    N = min(len(X_best), max_solutions)
    colors = matplotlib.cm.tab10.colors[:N]

    fig, axes = plt.subplots(N_VAR, 1, figsize=(15, N_VAR * 0.9), sharex=False)

    for i, ax in enumerate(axes):
        ticks = None
        if i == 0:
            ticks = NMOTORS_SET
        elif i == 7:
            ticks = VBATTERY_SET
        elif i == 8:
            ticks = CRATING_SET

        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 1, max(ticks) + 1)
            ax.hlines(0, min(ticks) - 1, max(ticks) + 1, color='gray', linewidth=5, alpha=0.3)
            for j in range(N):
                x_val = X_best[j, i]
                ax.plot(x_val, 0, 'o', color=colors[j], markersize=6)
        else:
            ax.hlines(0, XL[i], XU[i], color='gray', linewidth=5, alpha=0.3)
            ax.set_xlim(XL[i], XU[i])
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

def plot_parallel_coordinates(X, F, G, variable_names, max_solutions=5):
    N = max_solutions
    X_decoded = decode_discrete_vars(X)
    df_columns = variable_names + ["-T/W", "-FlightTime", "TotalMass"]
    data = np.hstack((X_decoded, F))
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
    feasible_indices = np.where(np.all(G <= 0, axis=1))[0]
    for i in range(min(len(feasible_indices), N)):
        idx = feasible_indices[i]
        df_norm.at[idx, "Solution"] = f"Solution {i+1}"

    plt.figure(figsize=(18, 8))
    parallel_coordinates(df_norm, class_column="Solution",
                        color='darkgray', alpha=0.2, linewidth=1)

    colors = matplotlib.cm.tab10.colors[:N]
    handles = []

    for i in range(N):
        sol_label = f"Solution {i+1}"
        parallel_coordinates(df_norm[df_norm['Solution'] == sol_label],
                            class_column="Solution",
                            color=[colors[i]], alpha=0.9, linewidth=2)

        handle = plt.Line2D([0], [0], color=colors[i], linewidth=2, label=sol_label)
        handles.append(handle)

    plt.ylabel("Normalized Values (min-max scaled)", fontsize=12)
    plt.title("Parallel Coordinates Plot - All Solutions with Top 5 Highlighted")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=90)
    plt.legend(handles=handles, loc='best')
    plt.tight_layout()
    plt.show()
