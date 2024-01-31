"""Module for RK4 integrator"""
from .base import Integrator, StateSpaceBase, ArrayLike, cs


class RK4(Integrator):
    @staticmethod
    def forward(
        state_space: StateSpaceBase,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        h: cs.SX | float,
    ):
        nq = state_space.model.nq
        nv = state_space.model.nv

        def state_dot(x, u):
            """return nv*2 vector of state derivative"""
            container = state_space.model.backend.math.zeros(2 * nv)
            container[:nv] = x[nq:]
            container[nv:] = state_space.model.forward_dynamics(x[:nq], x[nq:], u)

            return container.array

        def tangent_int(x, tang_x):
            """tangent integration of state by its increment"""

            # simplify notation for integration on manifold
            manifold_int = state_space.model.backend.integrate_configuration

            container = state_space.model.backend.math.zeros(nq + nv)

            # configuration is integrated on manifold using pinocchio implementation
            container[:nq] = manifold_int(h, x[:nq], tang_x[nv:])

            # velocity and acceleration are in the same space and do not require specific treatment
            container[nq:] = x[nq:] + h * tang_x[nv:]

            return container.array

        k1 = state_dot(x0, qfrc_u)
        k2 = state_dot(tangent_int(x0, 0.5 * h * k1), qfrc_u)
        k3 = state_dot(tangent_int(x0, 0.5 * h * k2), qfrc_u)
        k4 = state_dot(tangent_int(x0, h * k3), qfrc_u)

        return tangent_int(x0, (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
