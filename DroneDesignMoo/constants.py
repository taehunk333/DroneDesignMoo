# constants.py
import numpy as np

# Fixed Parameters
MASS_PAYLOAD = 1.5  # kg
AIR_DENSITY = 1.225  # kg/m^3
GRAV_CONSTANT = 9.81  # m/s^2
ENERGY_DENSITY = 200  # Wh/kg
ETA_MOTOR = 0.85
ETA_ESC = 0.95
KT = 5.5e-4 # empirical coefficient
MIN_FLIGHT_TIME = 0.5  # hours
ESC_VOLT_LIMIT = 22  # V
FRAME_DIAM_MAX = 0.83  # m

# Discrete sets
NUM_MOTORS_SET = [4, 6, 8]
BATT_VOLT_SET = [11.1, 14.8, 22.2]
C_RATE_SET = [25, 50, 75, 100]

# Variable bounds
XL = np.array([0, 200, 10, 0.05, 0.15, 3, 2000, 0, 0, 0.3])
XU = np.array([2, 1000, 90, 1, 0.8, 6, 30000, 2, 3, 1.5])

# Other constants
N_VAR = 10
N_OBJ = 3
N_CONSTR = 6
