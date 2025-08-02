# problem.py
import numpy as np
from pymoo.core.problem import Problem
from constants import *

class DroneOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=N_VAR, n_obj=N_OBJ, n_constr=N_CONSTR, xl=XL, xu=XU)

    def _evaluate(self, X, out, *args, **kwargs):
        nmotors_idx = np.clip(np.round(X[:, 0]).astype(int), 0, len(NMOTORS_SET) - 1)
        Vbattery_idx = np.clip(np.round(X[:, 7]).astype(int), 0, len(VBATTERY_SET) - 1)
        Crating_idx = np.clip(np.round(X[:, 8]).astype(int), 0, len(CRATING_SET) - 1)

        nmotors = np.array([NMOTORS_SET[i] for i in nmotors_idx])
        KV_motor = X[:, 1]
        Imotor_max = X[:, 2]
        mmotor = X[:, 3]
        Dprop = X[:, 4]
        Pprop = X[:, 5]
        Cbattery = X[:, 6]
        Vbattery = np.array([VBATTERY_SET[i] for i in Vbattery_idx])
        Crating = np.array([CRATING_SET[i] for i in Crating_idx])
        mframe = X[:, 9]

        mbattery = (Cbattery * Vbattery) / (EDENSITY * 1000)
        mtotal = MPAYLOAD + mframe + mbattery + nmotors * mmotor
        Aprop = (np.pi * Dprop ** 2) / 4

        Tmotor_max = KT * Dprop ** 3 * Pprop * KV_motor * Vbattery
        Phover = (mtotal * GRAVITY) ** 1.5 / np.sqrt(2 * RHO * nmotors * Aprop)
        tflight = (Cbattery * Vbattery * ETA_MOTOR * ETA_ESC) / (1000 * Phover)

        T_W = (nmotors * Tmotor_max) / (mtotal * GRAVITY)
        Idraw = (nmotors * Tmotor_max) / (ETA_MOTOR * Vbattery)
        Idraw_per_motor = Tmotor_max / (ETA_MOTOR * Vbattery)

        F1 = -T_W
        F2 = -tflight
        F3 = mtotal
        F = np.column_stack([F1, F2, F3])

        g1 = 2 * mtotal * GRAVITY - (nmotors * Tmotor_max)
        g2 = TREQUIRED - tflight
        g3 = Idraw - (Crating * Cbattery / 1000)
        g4 = Idraw_per_motor - Imotor_max
        g5 = Vbattery - VESC_MAX
        g6 = Dprop - DFRAME_MAX
        G = np.column_stack([g1, g2, g3, g4, g5, g6])

        out["F"] = F
        out["G"] = G
