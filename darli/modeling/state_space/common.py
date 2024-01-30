from ..base import ModelBase, StateSpaceBase
from ...backend import CasadiBackend
import casadi as cs
from typing import Dict
from ...arrays import ArrayLike
from ...quaternions import left_mult, expand_map
from ..integrators import Integrator, ForwardEuler


class StateSpace(StateSpaceBase):
    def __init__(self, model: ModelBase) -> None:
        self.__model: ModelBase = model

        self.__force_jacobians: Dict[str, ArrayLike] = {}

    @property
    def model(self):
        return self.__model

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

    def state_derivative(
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
            quat = quat / self.__model.backend.math.norm_2(quat).array

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
            container = self.__model.backend.math.zeros(
                self.__model.nv + self.__model.nv
            ).array
            container[: self.__model.nv] = self.__model.v
            container[self.__model.nv :] = self.__model.forward_dynamics()

            return container

    def rollout(
        self,
        state: ArrayLike,
        control: ArrayLike,
        dt: cs.SX | float,
        n_steps: int,
        integrator: Integrator = ForwardEuler,
    ) -> ArrayLike:
        """
        Rollout function propagates the state forward in time using the input and the state derivative

        Parameters
        ----------
        state: initial state
        input: input
        dt: time step
        n_steps: number of steps to propagate

        Returns
        -------
        state: propagated state
        """
        for _ in range(n_steps):
            state = integrator.forward(self, state, control, dt)

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
