# constants.py
import numpy as np

# Fixed Parameters
MASS_PAYLOAD = 1.5  # kg
AIR_DENSITY = 1.225  # kg/m^3
GRAV_CONSTANT = 9.81  # m/s^2
ENERGY_DENSITY = 200  # Wh/kg
ETA_MOTOR = 0.85
ETA_ESC = 0.95
KT = 0.5e-3  # empirical coefficient
TREQUIRED = 15 / 60  # hours
VESC_MAX = 25  # V
DFRAME_MAX = 0.6  # m

# Discrete sets
NMOTORS_SET = [4, 6, 8]
VBATTERY_SET = [11.1, 14.8, 22.2]
CRATING_SET = [25, 50, 75, 100]

# Variable bounds
XL = np.array([0, 500, 10, 0.05, 0.15, 3, 1000, 0, 0, 0.3])
XU = np.array([2, 2200, 60, 0.3, 0.5, 6, 20000, 2, 3, 2.5])

# Other constants
N_VAR = 10
N_OBJ = 3
N_CONSTR = 6
