"""Module for RK4 integrator"""
from .base import Integrator, StateSpaceBase, ArrayLike, cs
from typing import Callable


class RK4(Integrator):
    @staticmethod
    def forward(
        state_space: StateSpaceBase,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        dt: cs.SX | float,
    ):
        nq = state_space.model.nq

        # integrate velocity using runge-kutta method
        def xdot_func(x):
            return state_space.state_derivative(x[:nq], x[nq:], qfrc_u)

        return x0 + RK4.__step(x0, dt, xdot_func)

    @staticmethod
    def __step(y0, h, dydt: Callable[[ArrayLike, ArrayLike], ArrayLike]) -> ArrayLike:
        k1 = dydt(y0)
        k2 = dydt(y0 + 0.5 * h * k1)
        k3 = dydt(y0 + 0.5 * h * k2)
        k4 = dydt(y0 + h * k3)

        return (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
