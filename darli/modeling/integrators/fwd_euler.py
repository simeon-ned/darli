"""Module for forward euler integrator"""
from .base import Integrator, ModelBase, ArrayLike, cs


class ForwardEuler(Integrator):
    def __init__(self, model: ModelBase):
        super().__init__(model)

    def forward(
        self,
        derivative: ArrayLike,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        dt: cs.SX | float,
    ):

        log = derivative(x0[: self.nq], x0[self.nq :], qfrc_u)
        exp = self.tangent_step(x0, dt * log)

        return exp
