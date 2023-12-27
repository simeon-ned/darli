from darli.modeling import Robot, Functional
from darli.backend import PinocchioBackend, CasadiBackend
from robot_descriptions import z1_description
import numpy as np

if __name__ == "__main__":
    r1 = Robot(PinocchioBackend(z1_description.URDF_PATH))
    r2 = Functional(CasadiBackend(z1_description.URDF_PATH))

    print(r2.forward_dynamics())
    print(r1.forward_dynamics() - r2.forward_dynamics()["dv"])

    diff = r1.forward_dynamics() - r2.forward_dynamics()["dv"]

    # should be true
    print(np.allclose(diff, np.zeros_like(diff)))
