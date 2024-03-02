from . import CasadiStateSpace, StateSpaceBase

# from ..utils.arrays import ArrayLike
from ..model._base import ModelBase
import casadi as cs
from .integrators import Integrator
from typing import Type


class FunctionalStateSpace(StateSpaceBase):
    def __init__(
        self,
        model: ModelBase = None,
        space: CasadiStateSpace = None,
    ) -> None:
        if space is not None:
            self.__space = space
        else:
            self.__space = CasadiStateSpace(model.expression_model)

        # parameters should be set from the model is parametric, otherwise it should be empty
        self.__parameters = []
        self.__parameters_name = []

        if hasattr(model, "parameters"):
            self.__parameters = [model.parameters]
            self.__parameters_name = ["theta"]

    def set_integrator(self, integrator_cls: Type[Integrator]):
        """
        Set the integrator for the system using the provided Integrator class.

        Args:
            integrator_cls (Type[Integrator]): The class (constructor) of the integrator to be used.

        Returns:
            The result of setting the integrator in the underlying state space.
        """
        return self.__space.set_integrator(integrator_cls)

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
    def derivative(self):
        return cs.Function(
            "derivative",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
                *self.__parameters,
            ],
            [
                self.__space.derivative(
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
                *self.__parameters_name,
            ],
            ["derivative"],
        )

    @property
    def time_variation(self):
        return cs.Function(
            "time_variation",
            [
                self.__space.model.q,
                self.__space.model.v,
                self.__space.model.qfrc_u,
                *self.__space.model.contact_forces,
                *self.__parameters,
            ],
            [
                self.__space.time_variation(
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
                *self.__parameters_name,
            ],
            ["time_variation"],
        )

    def rollout(
        self, dt: float, n_steps: int, control_sampling: float | None = None
    ) -> cs.Function:
        if control_sampling is None:
            control_sampling = dt

        control_dim = int(n_steps * dt / control_sampling)
        container = self.__space.model.backend.math.zeros(
            (self.__space.model.nu, control_dim)
        ).array

        for i in range(control_dim):
            var = self.__space.model.backend.math.array(
                f"control_{i}", self.__space.model.nu
            ).array
            container[:, i] = var

        return cs.Function(
            "rollout",
            [
                self.__space.state,
                container,
                *self.__space.model.contact_forces,
                *self.__parameters,
            ],
            [
                self.__space.rollout(
                    self.__space.state, container, dt, n_steps, control_sampling
                )
            ],
            [
                "state",
                "controls",
                *self.__space.model.contact_names,
                *self.__parameters_name,
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
                *self.__parameters,
            ],
            [self.__space.state_jacobian],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
                *self.__parameters_name,
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
                *self.__parameters,
            ],
            [self.__space.input_jacobian],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
                *self.__parameters_name,
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
                *self.__parameters,
            ],
            [self.__space.force_jacobian(body_name)],
            [
                "q",
                "v",
                "tau",
                *self.__space.model.contact_names,
                *self.__parameters_name,
            ],
            ["input_jacobian"],
        )
