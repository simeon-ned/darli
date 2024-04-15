from ..model._base import ModelBase
from ._base import StateSpaceBase

import casadi as cs
from typing import Dict
from ..utils.arrays import ArrayLike
from ..utils.quaternions import left_mult, expand_map
from .integrators import Integrator, ForwardEuler


class CommonStateSpace(StateSpaceBase):
    def __init__(self, model: ModelBase) -> None:
        self.__model: ModelBase = model.expression_model

        self.__integrator: Integrator = ForwardEuler(self.__model)
        self.__force_jacobians: Dict[str, ArrayLike] = {}

    def set_integrator(self, integrator: Integrator):
        self.__integrator = integrator(self.__model)

    @property
    def model(self):
        return self.__model

    @property
    def integrator(self):
        return self.__integrator

    @property
    def force_jacobians(self):
        return self.__force_jacobians

    @property
    def state(self):
        container = self.__model.backend.math.zeros(
            self.__model.nq + self.__model.nv
        ).array
        container[: self.__model.nq] = self.__model.q
        container[self.__model.nq :] = self.__model.v

        return container

    def derivative(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        nv = self.__model.nv

        container = self.model.backend.math.zeros(2 * nv)
        container[: self.__model.nq] = q
        container[self.__model.nq :] = v

        return self.__integrator.derivative(container.array, u)

    def time_variation(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ):
        if q is not None or v is not None:
            self.__model.update(
                q if q is not None else self.__model.q,
                v if v is not None else self.__model.v,
                dv=None,
                u=u if u is not None else self.__model.qfrc_u,
            )

        if self.__model.nq != self.__model.nv:
            # free-flyer model case
            container = self.__model.backend.math.zeros(
                self.__model.nq + self.__model.nv
            ).array

            # extract velocity of the freebody
            container[:3] = self.__model.v[:3]
            # extract quaternion
            quat = self.__model.q[3:7]
            omega = self.__model.v[3:6]

            # normalize the quaternion to prevent drift over time
            # quat = quat / self.__model.backend.math.norm_2(quat).array

            # from scalar last to scalar first
            quat_changed = quat
            quat_changed[[0, 1, 2, 3]] = quat_changed[[3, 0, 1, 2]]

            quat_dot = (
                0.5
                * left_mult(quat_changed, self.__model.backend.math)
                @ expand_map(self.__model.backend.math)
                @ omega
            )

            # convert back to scalar last convention
            container[3:7] = quat_dot[[1, 2, 3, 0]]

            container[self.__model.nq :] = self.__model.forward_dynamics()

            return container
        else:
            # fixed-base case
            return self.derivative(q, v, u)

    def rollout(
        self,
        state: ArrayLike,
        controls: ArrayLike,
        dt: cs.SX | float,
        n_steps: int,
        control_sampling: cs.SX | float | None = None,
    ) -> ArrayLike:
        """
        Rollout function propagates the state forward in time using the input and the state derivative

        Parameters
        ----------
        state: initial state
        input: input
        dt: time step
        n_steps: number of steps to propagate
        control_sampling: time step for control, by default matches dt
        integrator: integrator to use, by default ForwardEuler

        Returns
        -------
        state: propagated state
        """
        if control_sampling is None:
            control_sampling = dt

        assert controls.shape[1] == int(
            n_steps * dt / control_sampling
        ), f"We expect controls to have shape[1] = {int(n_steps * dt / control_sampling)}, but got {controls.shape[1]}"

        time = 0
        control_i = 0
        control_time = 0

        for _ in range(n_steps):
            # take next control if time
            if control_time + control_sampling < time:
                control_i += 1
                control_time += control_sampling

            control = controls[:, control_i]
            state = self.__integrator.forward(state, control, dt)

            time += dt

        return state

    @property
    def state_jacobian(self):
        # should be implemented backend-wise
        raise NotImplementedError

    @property
    def input_jacobian(self):
        # should be implemented backend-wise
        raise NotImplementedError

    def force_jacobian(self, body_name: str) -> cs.Function:
        # should be implemented backend-wise
        raise NotImplementedError
