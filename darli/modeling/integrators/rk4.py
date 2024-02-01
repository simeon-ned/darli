"""Module for RK4 integrator"""
from .base import Integrator, StateSpaceBase, ArrayLike, cs


class RK4(Integrator):
    def __init__(self, state_space: StateSpaceBase):
        super().__init__(state_space)

    def forward(
        self,
        derivative: ArrayLike,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        dt: cs.SX | float,
    ):

        k1_log = derivative(x0[: self.nq], x0[self.nq :], qfrc_u)

        k2_exp = self.tangent_step(x0, 0.5 * dt * k1_log)
        k2_log = derivative(k2_exp[: self.nq], k2_exp[self.nq :], qfrc_u)

        k3_exp = self.tangent_step(x0, 0.5 * dt * k2_log)
        k3_log = derivative(k3_exp[: self.nq], k3_exp[self.nq :], qfrc_u)

        k4_exp = self.tangent_step(x0, dt * k3_log)
        k4_log = derivative(k4_exp[: self.nq], k4_exp[self.nq :], qfrc_u)

        return self.tangent_step(
            x0, (dt / 6.0) * (k1_log + 2 * k2_log + 2 * k3_log + k4_log)
        )
