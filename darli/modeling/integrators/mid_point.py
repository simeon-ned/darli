"""Module for forward euler integrator"""
from .base import Integrator, StateSpaceBase, ArrayLike, cs


class MidPoint(Integrator):
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

        return self.tangent_step(x0, dt * k2_log)
