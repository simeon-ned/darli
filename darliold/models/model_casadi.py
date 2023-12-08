from dataclasses import dataclass

from darliold.backends import CasadiBackend
from darliold.models.state_space import StateSpace
from darliold.models.body import Body
import casadi as cs
import numpy as np
from typing import List, Dict, Any
from darliold.models.model import RobotModel, CoM, Energy


@dataclass
class Quantity:
    fun: Any
    value: Any


class RobotModelCasadi(RobotModel):
    def __init__(self, urdf_path: str, bodies_names=None):
        super().__init__(urdf_path, CasadiBackend(urdf_path))

        # internal casadi variables to generate functions
        self._q = self._back._q
        self._v = self._back._v
        self._dv = self._back._dv
        self._tau = self._back._tau
        # self._nu = self.nv

        self.add_bodies(bodies_names)

        self.update_selector()

        # self.__state_space: StateSpace = StateSpace(model=self, update=True)

    # def joint_idx(self, joint_name: str) -> int:
    #     return self._back.joint_iq(joint_name)

    @property
    def inertia(self) -> cs.Function:
        return self._back.inertia_matrix

    @property
    def gravity(self) -> cs.Function:
        if not self._gravity:
            # compute gravity function and return it
            tau_grav = self._back.rnea(
                q=self._q, v=np.zeros(self.nv), dv=np.zeros(self.nv)
            )["tau"]

            self._gravity = Quantity(
                fun=cs.Function(
                    "gravity",
                    [self._q],
                    [tau_grav],
                    ["q"],
                    ["gravity"],
                ),
                value=tau_grav,
            )

        return self._gravity.fun

    @property
    def coriolis(self) -> cs.Function:
        if not self._coriolis:
            # compute coriolis function and return it
            tau_grav = self._back.rnea(
                q=self._q,
                v=np.zeros(self.nv),
                dv=np.zeros(self.nv),
            )["tau"]

            tau_bias = self._back.rnea(
                q=self._q,
                v=self._v,
                dv=np.zeros(self.nv),
            )["tau"]

            coriolis = tau_bias - tau_grav

            self._coriolis = Quantity(
                fun=cs.Function(
                    "coriolis",
                    [self._q, self._v],
                    [coriolis],
                    ["q", "v"],
                    ["coriolis"],
                ),
                value=coriolis,
            )

        return self._coriolis.fun

    @property
    def bias_force(self) -> cs.Function:
        if not self._bias_force:
            # compute bias force function and return it
            tau_bias = self._back.rnea(
                q=self._q,
                v=self._v,
                dv=np.zeros(self.nv),
            )["tau"]

            self._bias_force = Quantity(
                fun=cs.Function(
                    "bias_force",
                    [self._q, self._v],
                    [tau_bias],
                    ["q", "v"],
                    ["bias_force"],
                ),
                value=tau_bias,
            )

        return self._bias_force.fun

    @property
    def momentum(self) -> cs.Function:
        # TODO: implement me
        raise NotImplementedError

    @property
    def lagrangian(self) -> cs.Function:
        if not self._lagrangian:
            # compute lagrangian function and return it
            lagrangian = self._back.kinetic_energy(
                self._q, self._v
            ) - self._back.potential_energy(self._q)

            self._lagrangian = Quantity(
                fun=cs.Function(
                    "lagrangian",
                    [self._q, self._v],
                    [lagrangian],
                    ["q", "v"],
                    ["lagrangian"],
                ),
                value=lagrangian,
            )

        return self._lagrangian.fun

    @property
    def contact_forces(self) -> List[cs.Function]:
        if not self._contact_forces:
            self._contact_forces = []
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                self._contact_forces.append(body.contact._force)

        return self._contact_forces

    @property
    def contact_names(self) -> List[str]:
        if not self._contact_names:
            self._contact_names = []
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                self._contact_names.append(body.name)

        return self._contact_names

    @property
    def contact_qforce(self) -> cs.Function:
        if not self._contact_qforce:
            qforce_sum = 0
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                qforce_sum += body.contact.contact_qforce(self._q, body.contact._force)

            self._contact_qforce = Quantity(
                fun=cs.Function(
                    "contacts_qforce",
                    [self._q, *self.contact_forces],
                    [qforce_sum],
                    ["q", *self.contact_names],
                    ["contacts_qforce"],
                ),
                value=qforce_sum,
            )

        return self._contact_qforce.fun

    @property
    def coriolis_matrix(self) -> cs.Function:
        if not self._coriolis_matrix:
            coriolis = self.coriolis(self._q, self._v)
            corilois_matrix = cs.jacobian(coriolis, self._v)

            self._coriolis_matrix = Quantity(
                fun=cs.Function(
                    "coriolis_matrix",
                    [self._q, self._v],
                    [corilois_matrix],
                    ["q", "v"],
                    ["coriolis_matrix"],
                ),
                value=corilois_matrix,
            )

        return self._coriolis_matrix.fun

    @property
    def forward_dynamics(self) -> cs.Function:
        if not self._forward_dynamics:
            # TODO: better to wrap it once more
            self.contact_qforce
            qforce_sum = self._contact_qforce.value

            dv = self._back.aba(
                q=self._q,
                v=self._v,
                tau=self._qfrc_u + qforce_sum,
            )["dv"]

            self._forward_dynamics = Quantity(
                fun=cs.Function(
                    "forward_dynamics",
                    [self._q, self._v, self._u, *self.contact_forces],
                    [dv],
                    ["q", "v", "u", *self.contact_names],
                    ["dv"],
                ),
                value=dv,
            )

        return self._forward_dynamics.fun

    @property
    def inverse_dynamics(self) -> cs.Function:
        if not self._inverse_dynamics:
            # TODO: better to wrap it once more
            self.contact_qforce
            qforce_sum = self._contact_qforce.value

            tau = (
                self._back.rnea(
                    q=self._q,
                    v=self._v,
                    dv=self._dv,
                )["tau"]
                - qforce_sum
            )

            self._inverse_dynamics = Quantity(
                fun=cs.Function(
                    "inverse_dynamics",
                    [self._q, self._v, self._dv, *self.contact_forces],
                    [tau],
                    ["q", "v", "dv", *self.contact_names],
                    ["tau"],
                ),
                value=tau,
            )

        return self._inverse_dynamics.fun

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    # @property
    # def selector(self):
    #     return self._selector

    def update_selector(
        self, matrix: np.ndarray | None = None, passive_joints: List[str] | None = None
    ):
        super().update_selector(matrix, passive_joints)
        # self._selector = np.eye(self.nv)

        # if matrix is not None:
        #     self._selector = matrix
        #     self.nu = np.shape(self._selector)[1]

        # if passive_joints is not None and matrix is None:
        #     joint_id = []
        #     self.nu = self.nv - len(passive_joints)
        #     for joint in passive_joints:
        #         if isinstance(joint, str):
        #             joint_id.append(self.joint_idx(joint))
        #         if isinstance(joint, int):
        #             joint_id.append(joint)
        #     self._selector = np.delete(self._selector, joint_id, axis=1)

        self._u = cs.SX.sym("u", self.nu)
        self._qfrc_u = cs.mtimes(self.selector, self._u)

        # unset what we have to compute again due to changes
        self._forward_dynamics = None

    # def add_bodies(self, bodies_names: List[str] | Dict[str, str]):
    #     """Adds bodies to the model and update the model"""
    #     if not bodies_names or len(bodies_names) == 0:
    #         return

    #     if isinstance(bodies_names, dict):
    #         self.bodies_names = bodies_names
    #         for body_pairs in self.bodies_names.items():
    #             body = Body(name=dict([body_pairs]), kindyn_backend=self._back)
    #             self.bodies[body_pairs[0]] = body
    #     elif isinstance(bodies_names, list):
    #         self.bodies_names = set(bodies_names)
    #         for body_name in self.bodies_names:
    #             body = Body(name=body_name, kindyn_backend=self._back)
    #             self.bodies[body_name] = body
    #     else:
    #         raise TypeError(
    #             f"unknown type of mapping is passed to add bodies: {type(bodies_names)}"
    #         )

    #     # update what we have to compute again due to changes
    #     self.__contact_forces = None
    #     self.__contact_names = None
    #     self.__contact_qforce = None

    #     # self.state_space.update()

    # def body(self, body_name) -> Body:
    #     try:
    #         return self.bodies[body_name]
    #     except KeyError:
    #         raise KeyError(f"Body {body_name} is not added")
