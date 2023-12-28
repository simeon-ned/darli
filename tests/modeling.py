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

    def test_energy(self):
        q = np.random.randn(self.robot.nq)
        dq = np.random.randn(self.robot.nv)

        # kinetic energy test
        kin = self.robot.energy(q, dq).kinetic
        kin_funct = self.funct.energy.kinetic(q, dq)

        self.assertTrue(np.allclose(kin - kin_funct, 0))

        # potential energy test
        pot = self.robot.energy(q).potential
        pot_funct = self.funct.energy.potential(q)

        self.assertTrue(np.allclose(pot - pot_funct, 0))

    def inertia(self):
        q = np.random.randn(self.robot.nq)

        # inertia test
        M = self.robot.inertia(q)
        M_funct = self.funct.inertia(q)

        self.assertTrue(np.allclose(M - M_funct, 0))

    def test_coriolis(self):
        q = np.random.randn(self.robot.nq)
        dq = np.random.randn(self.robot.nv)

        # coriolis test
        C = self.robot.coriolis(q, dq)
        C_funct = self.funct.coriolis(q, dq)

        self.assertTrue(np.allclose(C - C_funct, 0))

    def test_bias_force(self):
        q = np.random.randn(self.robot.nq)
        dq = np.random.randn(self.robot.nv)

        # bias force test
        b = self.robot.bias_force(q, dq)
        b_funct = self.funct.bias_force(q, dq)

        self.assertTrue(np.allclose(b - b_funct, 0))


if __name__ == "__main__":
    unittest.main()
