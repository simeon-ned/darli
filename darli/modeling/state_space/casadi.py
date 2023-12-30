from .common import StateSpace
from ..base import ModelBase
import casadi as cs
from ...backend import CasadiBackend


class CasadiStateSpace(StateSpace):
    def __init__(self, model: ModelBase) -> None:
        super().__init__(model)

        assert isinstance(model.backend, CasadiBackend), "This is Casadi-only class"

    @property
    def state_jacobian(self):
        return cs.jacobian(self.state_derivative, self.state)

    @property
    def input_jacobian(self):
        return cs.jacobian(self.state_derivative, self.model.qfrc_u)

    def force_jacobian(self, body_name: str) -> cs.Function:
        # early quit if we have already computed the jacobian
        if body_name in self.force_jacobians:
            return self.force_jacobians[body_name]

        if any(body_name in body.name for body in self.model.bodies.values()):
            raise KeyError(f"Body {body_name} is not added to the model")

        if self.bodies[body_name].contact is None:
            raise KeyError(f"Body {body_name} has no contact")

        xdot = self.state_derivative
        body = self.model.body(body_name)

        if body_name not in self.force_jacobians:
            self.force_jacobians[body_name] = cs.jacobian(xdot, body.contact.force)

        return self.force_jacobians[body_name]
