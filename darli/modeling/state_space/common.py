from ..base import ModelBase, StateSpaceBase
from ...backend import CasadiBackend
import casadi as cs
from typing import Dict
from ...arrays import ArrayLike
from ...quaternions import L, H


class StateSpace(StateSpaceBase):
    def __init__(self, model: ModelBase) -> None:
        self.__model = model

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

    @property
    def state_derivative(self):
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
                * L(quat_changed, self.__model.backend.math)
                @ H(self.__model.backend.math)
                @ omega
            )

            # convert back to scalar last convention
            container[3:7] = quat_dot[[1, 2, 3, 0]]

            container[self.__model.nq :] = self.__model.forward_dynamics(
                self.__model.q, self.__model.v, self.__model.qfrc_u
            )

            return container
        else:
            # fixed-base case
            container = self.__model.backend.math.zeros(
                self.__model.nv + self.__model.nv
            ).array
            container[: self.__model.nv] = self.__model.v
            container[self.__model.nv :] = self.__model.forward_dynamics(
                self.__model.q, self.__model.v, self.__model.qfrc_u
            )

            return container

    @property
    def state_jacobian(self):
        raise NotImplementedError
        # ensure we use casadi backend
        assert isinstance(
            self.__model.backend, CasadiBackend
        ), "Only casadi backend is supported for now"

        return cs.jacobian(self.state_derivative, self.state)

    @property
    def input_jacobian(self):
        raise NotImplementedError
        # ensure we use casadi backend
        assert isinstance(
            self.__model.backend, CasadiBackend
        ), "Only casadi backend is supported for now"

        return cs.jacobian(self.state_derivative, self.__model.qfrc_u)

    def force_jacobian(self, body_name: str) -> cs.Function:
        raise NotImplementedError
        # ensure we use casadi backend
        assert isinstance(
            self.__model.backend, CasadiBackend
        ), "Only casadi backend is supported for now"

        # early quit if we have already computed the jacobian
        if body_name in self.__force_jacobians:
            return self.__force_jacobians[body_name]

        if any(body_name in body.name for body in self.__model.bodies.values()):
            raise KeyError(f"Body {body_name} is not added to the model")

        if self.bodies[body_name].contact is None:
            raise KeyError(f"Body {body_name} has no contact")

        self.state_derivative
        xdot = self.__state_derivative.value

        body = self.__model.body(body_name)

        if body_name not in self.__force_jacobians:
            self.__force_jacobians[body_name] = cs.jacobian(xdot, body.contact.force)

        return self.__force_jacobians[body_name]
