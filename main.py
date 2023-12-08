from darli.backend import PinocchioBackend as Backend
# from darli.backend import CasadiBackend as Backend
from darli.backend import BackendBase as BackendBase
from darli.modeling import Robot, Symbolic
from robot_descriptions import z1_description
import numpy as np
import casadi as cs

if __name__ == "__main__":
    back: BackendBase = Backend(z1_description.URDF_PATH)

    back.inertia_matrix()

    model: Robot = Robot(back)

    res = model.gravity(np.array([0, 0, 0, 0, 0, 0]))
    print("-" * 30)
    print(res)

    back.update(q=np.array([1, 1, 1, 1, 1, 1]), v=np.zeros(6), dv=np.zeros(6))

    # res = model.gravity(cs.SX.sym("symq", 6))
    res = model.gravity()
    print("-" * 30)
    print(res)

    res = model.gravity(np.array([3, 1, 1, 3, 1, 1]))
    print("-" * 30)
    print(res)

    sym = Symbolic(back)
    print("-" * 30)
    print(sym.gravity())
