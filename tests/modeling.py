import unittest

from darli.modeling import Robot
from darli.modeling.functional import Functional
from darli.backend import PinocchioBackend, CasadiBackend
from robot_descriptions import z1_description
import numpy as np


class TestRobot(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(PinocchioBackend(z1_description.URDF_PATH))
        self.funct = Functional(CasadiBackend(z1_description.URDF_PATH))

    def test_com(self):
        q = np.random.randn(self.robot.nq)
        dq = np.random.randn(self.robot.nv)
        ddq = np.random.randn(self.robot.nv)

        # position test
        com = self.robot.com(q).position
        com_funct = self.funct.com.position(q)

        self.assertTrue(np.allclose(com - com_funct, 0))

        # velocity test
        com = self.robot.com(q, dq).velocity
        com_funct = self.funct.com.velocity(q, dq)

        self.assertTrue(np.allclose(com - com_funct, 0))

        # acceleration test
        com = self.robot.com(q, dq, ddq).acceleration
        com_funct = self.funct.com.acceleration(q, dq, ddq)

        self.assertTrue(np.allclose(com - com_funct, 0))

        # jacobian test
        jac = self.robot.com(q).jacobian
        jac_funct = self.funct.com.jacobian(q)

        self.assertTrue(np.allclose(jac - jac_funct, 0))


if __name__ == "__main__":
    unittest.main()
