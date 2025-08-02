import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination

# ---------------- Fixed Parameters ----------------
mpayload = 1.5  # kg
rho = 1.225  # kg/m^3
g = 9.81  # m/s^2
Edensity = 200  # Wh/kg
eta_motor = 0.85
eta_ESC = 0.95
kT = 1e-3  # empirical coefficient (adjust as needed)
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

# Treat discrete variables as continuous indices
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
    # --------------------------------

    print(f"Generation {algorithm.n_gen}")

plt.ioff()
plt.show()

# ---------------- Final Results ----------------
res = algorithm.pop
F = res.get("F")
print("hash", F.sum())
