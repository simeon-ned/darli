from ..state_space import CasadiStateSpace
from ..base import ModelBase, StateSpaceBase, ArrayLike
import casadi as cs
from ..integrators import Integrator, ForwardEuler


class FunctionalStateSpace(StateSpaceBase):
    def __init__(
        self,
        model: ModelBase = None,
        space: CasadiStateSpace = None,
    ) -> None:
        if space is not None:
            self.__space = space
        else:
            self.__space = CasadiStateSpace(model)

    @classmethod
    def from_space(cls, space: CasadiStateSpace):
        return cls(space=space)

    @property
    def model(self):
        return self.__space.model

    @property
    def force_jacobians(self):
        return self.__space.force_jacobians

    @property
    def state(self):
        return cs.Function(
            "state",
            [self.__space.model.q, self.__space.model.v],
            [self.__space.state],
        )

    @property
    def state_derivative(self):
        return cs.Function(
            "state_derivative",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
            ],
            [
                self.__space.state_derivative(
                    self.__space.model.q,
                    self.__space.model.v,
                    self.__space.model.qfrc_u,
                )
            ],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
            ],
            ["state_derivative"],
        )

    def rollout(
        self,
        dt: cs.SX,
        n_steps: cs.SX,
        integrator: Integrator = ForwardEuler,
    ) -> cs.Function:
        return cs.Function(
            "rollout",
            [
                self.__space.state,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
                dt,
                n_steps,
            ],
            [
                self.__space.rollout(
                    self.__space.state,
                    self.__space.__model.qfrc_u,
                    dt,
                    n_steps,
                    integrator,
                )
            ],
            [
                "state",
                "tau",
                *self.__space.model.contact_names,
                "dt",
                "n_steps",
            ],
            ["next_state"],
        )

    @property
    def state_jacobian(self):
        return cs.Function(
            "state_jacobian",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
            ],
            [self.__space.state_jacobian],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
            ],
            ["state_jacobian"],
        )

    @property
    def input_jacobian(self):
        return cs.Function(
            "input_jacobian",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
            ],
            [self.__space.input_jacobian],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
            ],
            ["input_jacobian"],
        )

    def force_jacobian(self, body_name: str) -> cs.Function:
        return cs.Function(
            f"force_jacobian_{body_name}",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
            ],
            [self.__space.force_jacobian(body_name)],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
            ],
            ["input_jacobian"],
        )
