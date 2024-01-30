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
        nv = state_space.model.nv

        q, v = x0[:nq], x0[nq:]

        # integrate velocity using runge-kutta method
        def vdot_func(v_n):
            return state_space.model.forward_dynamics(q, v_n, qfrc_u)

        integrated_v = v + RK4.__step(v, dt, vdot_func)

        # integrate configuration using pinocchio fancy lie-group
        # TODO: it is not RK4 on manifold
        integrated_q = state_space.model.backend.integrate_configuration(
            dt, q, integrated_v
        )

        container = state_space.model.backend.math.zeros(nq + nv).array
        container[:nq] = integrated_q
        container[nq:] = integrated_v

        return container

    @staticmethod
    def __step(y0, h, dydt: Callable[[ArrayLike, ArrayLike], ArrayLike]) -> ArrayLike:
        k1 = dydt(y0)
        k2 = dydt(y0 + 0.5 * h * k1)
        k3 = dydt(y0 + 0.5 * h * k2)
        k4 = dydt(y0 + h * k3)

        return (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
