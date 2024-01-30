"""Module for forward euler integrator"""
from .base import Integrator, StateSpaceBase, ArrayLike, cs


class ForwardEuler(Integrator):
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
        vdot = state_space.model.forward_dynamics(q, v, qfrc_u)

        # forward euler to integrate velocity
        integrated_v = v + dt * vdot

        # pinocchio fancy lie-group integration
        integrated_q = state_space.model.backend.integrate_configuration(
            dt, q, integrated_v
        )

        container = state_space.model.backend.math.zeros(nq + nv).array
        container[:nq] = integrated_q
        container[nq:] = integrated_v

        return container
