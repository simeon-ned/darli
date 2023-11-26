import casadi as cs
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import numpy as np
from symbotics.utils import RecursiveNamespace

# BUG:
# THE DICTIONARY OF JACOBIANS FOR BODIES AND VELOCITIES ARE REPEATED


# TODO:
# all necesary functions are now depend on parameters


# regressors

# //////
# Bodies
# regressor.momentum
# regressor.wrench
# regressor.energy.potential
# regressor.energy.kinetic
