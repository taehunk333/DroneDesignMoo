import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# Fixed Parameters
MASS_PAYLOAD = 1.5  # kg
AIR_DENSITY = 1.225  # kg/m3
GRAV_CONSTANT = 9.81  # m/s2
ENERGY_DENSITY = 200  # Wh/kg
EFF_MOTOR = 0.85
EFF_ESC = 0.95
V_ESC_MAX = 25  # V
DIA_FRAME_MAX = 1  # m
FLIGHT_TIME_REQ = 10  # Required flight time in minutes

# Discrete Variable Maps
discrete_map = {
    'n_motors': [4],
    'V_battery': [11.1, 14.8, 22.2],
    'C_rating': [25, 50, 75, 100]
}

class DroneOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=10,
            n_obj=3,
            n_constr=6,
            xl=np.array([0, 500, 10, 0.05, 0.15, 3, 1000, 0, 0, 0.3]),  # P_prop bounds [3,6]
            xu=np.array([2, 2200, 60, 0.3, 0.5, 6, 20000, 2, 3, 2.5]),
            elementwise_evaluation=True
        )

    def _evaluate(self, x, out, *args, **kwargs):
        idx_n_motors = int(np.clip(round(x[0]), 0, len(discrete_map['n_motors']) - 1))
        idx_V_battery = int(np.clip(round(x[7]), 0, len(discrete_map['V_battery']) - 1))
        idx_C_rating = int(np.clip(round(x[8]), 0, len(discrete_map['C_rating']) - 1))

        n_motors = discrete_map['n_motors'][idx_n_motors]
        KV_motor = x[1]
        I_motor_max = x[2]
        m_motor = x[3]
        D_prop = x[4]
        P_prop = x[5]  # continuous pitch now
        C_battery = x[6]
        V_battery = discrete_map['V_battery'][idx_V_battery]
        C_rating = discrete_map['C_rating'][idx_C_rating]
        m_frame = x[9]

        m_battery = (C_battery / 1000 * V_battery) / ENERGY_DENSITY
        m_total = MASS_PAYLOAD + m_frame + m_battery + n_motors * m_motor
        A_prop = np.pi * (D_prop / 2) ** 2

        k_T = 10
        T_motor_max = k_T * D_prop ** 3 * P_prop * KV_motor * V_battery

        P_hover = (m_total * GRAV_CONSTANT) ** 1.5 / np.sqrt(2 * AIR_DENSITY * n_motors * A_prop)

        t_flight = (C_battery * V_battery * EFF_MOTOR * EFF_ESC) / (1000 * P_hover) / 60  # minutes

        TWR = (n_motors * T_motor_max) / (m_total * GRAV_CONSTANT)

        # Objectives (minimize negative for maximization)
        f1 = -TWR
        f2 = -t_flight
        f3 = m_total

        # Constraints (g(x) <= 0)
        g1 = 2 * m_total * GRAV_CONSTANT - n_motors * T_motor_max  # thrust margin
        g2 = FLIGHT_TIME_REQ - t_flight  # flight time req
        g3 = (n_motors * T_motor_max) / (EFF_MOTOR * V_battery) - (C_rating * C_battery / 1000)  # battery current
        g4 = (T_motor_max) / (EFF_MOTOR * V_battery) - I_motor_max  # motor current
        g5 = V_battery - V_ESC_MAX  # ESC voltage limit
        g6 = D_prop - DIA_FRAME_MAX  # frame size

        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2, g3, g4, g5, g6]

# Callback to track best objectives per generation
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best_F"] = []

    def notify(self, algorithm):
        # Save best (minimum) objective vector from current population
        best_F = algorithm.pop.get("F").min(axis=0)
        self.data["best_F"].append(best_F)

if __name__ == "__main__":
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    problem = DroneOptimizationProblem()

    algorithm = NSGA3(
        pop_size=100,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)

    callback = MyCallback()

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True,
                   callback=callback)

    print("\nBest Solutions Found:")
    for i in range(min(5, len(res.X))):
        x = res.X[i]
        idx_n_motors = int(np.clip(round(x[0]), 0, len(discrete_map['n_motors']) - 1))
        idx_V_battery = int(np.clip(round(x[7]), 0, len(discrete_map['V_battery']) - 1))
        idx_C_rating = int(np.clip(round(x[8]), 0, len(discrete_map['C_rating']) - 1))

        n_motors = discrete_map['n_motors'][idx_n_motors]
        KV_motor = x[1]
        I_motor_max = x[2]
        m_motor = x[3]
        D_prop = x[4]
        P_prop = x[5]  # continuous
        C_battery = x[6]
        V_battery = discrete_map['V_battery'][idx_V_battery]
        C_rating = discrete_map['C_rating'][idx_C_rating]
        m_frame = x[9]

        # Note: objectives stored as negatives for TWR and flight time
        TWR = -res.F[i][0]
        t_flight = -res.F[i][1]
        m_total = res.F[i][2]

        print(f"Solution {i+1}:")
        print(f"  n_motors: {n_motors}")
        print(f"  KV_motor: {KV_motor:.2f} RPM/V")
        print(f"  I_motor_max: {I_motor_max:.2f} A")
        print(f"  m_motor: {m_motor:.3f} kg")
        print(f"  D_prop: {D_prop:.3f} m")
        print(f"  P_prop: {P_prop:.3f} inches")
        print(f"  C_battery: {C_battery:.1f} mAh")
        print(f"  V_battery: {V_battery} V")
        print(f"  C_rating: {C_rating}")
        print(f"  m_frame: {m_frame:.3f} kg")
        print(f"  Objectives (TWR, t_flight, m_total): {TWR:.3f}, {t_flight:.3f}, {m_total:.3f}")
        print()

    # Plot convergence of best objectives per generation
    best_F = np.array(callback.data["best_F"])
    generations = np.arange(1, best_F.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, -best_F[:, 0], label="Best TWR (max)")
    plt.plot(generations, -best_F[:, 1], label="Best Flight Time (max)")
    plt.plot(generations, best_F[:, 2], label="Best Total Mass (min)")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.title("Convergence of Best Objective Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
