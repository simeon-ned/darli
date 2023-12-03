import casadi as cs
from dataclasses import dataclass
from typing import Dict

# TODO:

# Features:
# observation mapping
# momentum state space
# velocity mapping v = G(q)dv
# quaternion derivatives
# "kinematic state"
# use forward_dynamics
# baumgarte stabilization
# proper jacobians with respect to quaternions
# integrator


@dataclass
class Quantity:
    fun: cs.Function | None
    value: cs.SX | float


class StateSpace:
    def __init__(self, name=None, model=None, velocity_mapping=None, update=False):
        if model is None:
            print("Provide the symbotics model as input...")
            return
        else:
            self.robot_model = model

        self._q = self.robot_model._q
        self._v = self.robot_model._v
        self._u = self.robot_model._u
        self._state = cs.vertcat(self._q, self._v)

        self.contact_forces = self.robot_model.contact_forces
        self.contact_names = self.robot_model.contact_names
        self.bodies = self.robot_model.bodies
        self.forward_dynamics = self.robot_model.forward_dynamics

        self.state = None
        self.state_derivative = None
        self.state_jacobian = None
        self.input_jacobian = None
        self.velocity_mapping = None
        self.integrator = None

        self._force_jacobian = {}

        if update:
            self.update()

        self.__state_derivative: Quantity | None = None
        self.__state_jacobian: Quantity | None = None
        self.__input_jacobian: Quantity | None = None
        self.__force_jacobians: Dict[str, Quantity] = {}

    @property
    def state_derivative(self):
        if not self.__state_derivative:
            dv = self.forward_dynamics(self._q, self._v, self._u, *self.contact_forces)
            xdot = cs.vertcat(self._v, dv)

            self.__state_derivative = Quantity(
                cs.Function(
                    "state_derivative",
                    [self._state, self._u, *self.contact_forces],
                    [xdot],
                    ["x", "u", *self.contact_names],
                    ["dxdt"],
                ),
                xdot,
            )

        return self.__state_derivative.fun

    @property
    def state_jacobian(self):
        if not self.__state_jacobian:
            # TODO: ensure it is computed
            self.state_derivative
            xdot = self.__state_derivative.value

            dfdx = cs.jacobian(xdot, self._state)
            self.__state_jacobian = Quantity(
                cs.Function(
                    "state_jacobian",
                    [self._state, self._u, *self.contact_forces],
                    [dfdx],
                    ["x", "u", *self.contact_names],
                    ["dfdx"],
                ),
                dfdx,
            )

        return self.__state_jacobian.fun

    @property
    def input_jacobian(self):
        if not self.__input_jacobian:
            # TODO: ensure it is computed
            self.state_derivative
            xdot = self.__state_derivative.value

            dfdu = cs.jacobian(xdot, self._u)
            self.__input_jacobian = Quantity(
                cs.Function(
                    "input_jacobian",
                    [self._state, self._u, *self.contact_forces],
                    [dfdu],
                    ["x", "u", *self.contact_names],
                    ["dfdu"],
                ),
                dfdu,
            )

        return self.__input_jacobian.fun

    def update(self):
        self.state_derivative
        xdot = self.__state_derivative.value

        for body in self.bodies.values():
            if body.contact is not None:
                df_dforce = cs.jacobian(xdot, body.contact._force)
                force_jacobian = cs.Function(
                    f"{body.name}_jacobian",
                    [self._state, self._u, *self.contact_forces],
                    [df_dforce],
                    ["x", "u", *self.contact_names],
                    ["dfdforce"],
                )
                self._force_jacobian[body.name] = force_jacobian

    def force_jacobian(self, body_name: str) -> cs.Function:
        if any(body_name in body.name for body in self.bodies.values()):
            raise KeyError(f"Body {body_name} is not added to the model")

        if self.bodies[body_name].contact is None:
            raise KeyError(f"Body {body_name} has no contact")

        self.state_derivative
        xdot = self.__state_derivative.value

        body = self.bodies[body_name]

        if body_name not in self.__force_jacobians:
            df_dforce = cs.jacobian(xdot, body.contact._force)
            force_jacobian = cs.Function(
                f"{body.name}_jacobian",
                [self._state, self._u, *self.contact_forces],
                [df_dforce],
                ["x", "u", *self.contact_names],
                ["dfdforce"],
            )
            self.__force_jacobians[body_name] = force_jacobian

        try:
            return self.__force_jacobian[body_name]
        except KeyError:
            raise KeyError(f"Body {body_name} is not added or has no contact")
