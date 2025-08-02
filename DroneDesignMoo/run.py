import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

# Fixed Parameters
m_payload = 1.5  # kg
air_density = 1.225  # kg/m3
g = 9.81  # m/s2
E_density = 200  # Wh/kg
eff_motor = 0.85
eff_ESC = 0.95
V_ESC_max = 25  # V
D_frame_max = 1  # m
t_required = 10  # Required flight time in minutes

# Discrete Variable Maps
discrete_map = {
    'n_motors': [4, 6, 8],
    'P_prop': [3, 4, 5, 6],
    'V_battery': [11.1, 14.8, 22.2],
    'C_rating': [25, 50, 75, 100]
}

class DroneOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=10,
                         n_obj=3,
                         n_constr=6,
                         xl=np.array([0, 500, 10, 0.05, 0.15, 0, 1000, 0, 0, 0.3]),
                         xu=np.array([2, 2200, 60, 0.3, 0.5, 3, 20000, 2, 3, 2.5]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Decode discrete variables safely with clipping
        n_motors = discrete_map['n_motors'][int(np.clip(round(x[0]), 0, len(discrete_map['n_motors']) - 1))]
        KV_motor = x[1]
        I_motor_max = x[2]
        m_motor = x[3]
        D_prop = x[4]
        P_prop = discrete_map['P_prop'][int(np.clip(round(x[5]), 0, len(discrete_map['P_prop']) - 1))]
        C_battery = x[6]
        V_battery = discrete_map['V_battery'][int(np.clip(round(x[7]), 0, len(discrete_map['V_battery']) - 1))]
        C_rating = discrete_map['C_rating'][int(np.clip(round(x[8]), 0, len(discrete_map['C_rating']) - 1))]
        m_frame = x[9]

        # Derived Quantities
        m_battery = (C_battery / 1000 * V_battery) / E_density
        m_total = m_payload + m_frame + m_battery + n_motors * m_motor
        A_prop = np.pi * (D_prop / 2) ** 2

        k_T = 10  # Simplified thrust coefficient
        T_motor_max = k_T * D_prop ** 3 * P_prop * KV_motor * V_battery

        P_hover = (m_total * g) ** 1.5 / np.sqrt(2 * air_density * n_motors * A_prop)

        t_flight = (C_battery * V_battery * eff_motor * eff_ESC) / (1000 * P_hover) / 60  # in minutes

        TWR = (n_motors * T_motor_max) / (m_total * g)

        # Objectives (minimize negative for maximization)
        f1 = -TWR
        f2 = -t_flight
        f3 = m_total

        # Constraints (g(x) <= 0)
        g1 = 2 * m_total * g - n_motors * T_motor_max  # Thrust margin
        g2 = t_required - t_flight  # Flight time requirement
        g3 = (n_motors * T_motor_max) / (eff_motor * V_battery) - (C_rating * C_battery / 1000)  # Battery current
        g4 = (T_motor_max) / (eff_motor * V_battery) - I_motor_max  # Motor current
        g5 = V_battery - V_ESC_max  # ESC voltage limit
        g6 = D_prop - D_frame_max  # Frame size constraint

        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2, g3, g4, g5, g6]


# Reference Directions for 3 objectives
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

# Enable returning constraint values 'G' along with objectives and decision variables
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True,
               return_values_of=["F", "G", "X"])


# Check if constraint values are returned
if res.G is not None:
    feasible_mask = np.all(res.G <= 0, axis=1)
    n_feasible = np.sum(feasible_mask)
    print(f"\nTotal feasible solutions found: {n_feasible}\n")
else:
    print("\nConstraint values (G) were not returned.\n")
    feasible_mask = None

# Extract feasible solutions from res.opt (pymoo stores feasible opt points here)
if res.opt is not None and len(res.opt) > 0:
    print(f"Number of feasible solutions in Pareto Front: {len(res.opt)}")
    print("Best Feasible Solutions Found:")
    for i, sol in enumerate(res.opt[:5]):  # Show up to 5 feasible solutions
        x = sol.get("X")
        F = sol.get("F")
        # Decode discrete variables with clipping
        n_motors = discrete_map['n_motors'][int(np.clip(round(x[0]), 0, len(discrete_map['n_motors']) - 1))]
        KV_motor = x[1]
        I_motor_max = x[2]
        m_motor = x[3]
        D_prop = x[4]
        P_prop = discrete_map['P_prop'][int(np.clip(round(x[5]), 0, len(discrete_map['P_prop']) - 1))]
        C_battery = x[6]
        V_battery = discrete_map['V_battery'][int(np.clip(round(x[7]), 0, len(discrete_map['V_battery']) - 1))]
        C_rating = discrete_map['C_rating'][int(np.clip(round(x[8]), 0, len(discrete_map['C_rating']) - 1))]
        m_frame = x[9]

        print(f"Solution {i + 1}:")
        print(f"  n_motors: {n_motors}")
        print(f"  KV_motor: {KV_motor:.2f} RPM/V")
        print(f"  I_motor_max: {I_motor_max:.2f} A")
        print(f"  m_motor: {m_motor:.3f} kg")
        print(f"  D_prop: {D_prop:.3f} m")
        print(f"  P_prop: {P_prop} inches")
        print(f"  C_battery: {C_battery:.1f} mAh")
        print(f"  V_battery: {V_battery} V")
        print(f"  C_rating: {C_rating}")
        print(f"  m_frame: {m_frame:.3f} kg")
        print(f"  Objectives (TWR, t_flight, m_total): {F}")
        print()
else:
    print("No feasible solutions found in Pareto Front.")

