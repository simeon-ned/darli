from dataclasses import dataclass

from darliold.backends import PinocchioBackend
from darliold.models.state_space import StateSpace
from darliold.models.body import Body
import numpy as np
from typing import List, Dict
from darliold.models.model import RobotModel
import pinocchio as pin
from numpy.typing import ArrayLike


@dataclass
class CoM:
    position: ArrayLike
    velocity: ArrayLike
    acceleration: ArrayLike
    jacobian: ArrayLike
    jacobian_dt: ArrayLike


@dataclass
class Energy:
    kinetic: float
    potential: float


class RobotModelPinocchio(RobotModel):
    def __init__(self, urdf_path: str, bodies_names=None):
        super().__init__(urdf_path, PinocchioBackend(urdf_path))

        self._q = self._back._q
        self._v = self._back._v
        self._dv = self._back._dv
        self._tau = self._back._tau
        # self._nu = self.nv

        self.add_bodies(bodies_names)

        # self.update_selector()

        # self.joint_map: Dict[str, int] = dict()
        # for joint_name in self.joint_names:
        #     self.joint_map[joint_name] = self.joint_idx(joint_name)

        # self.__state_space: StateSpace = StateSpace(model=self, update=True)

    def update(
        self, q: ArrayLike, v: ArrayLike, dv: ArrayLike = None, tau: ArrayLike = None
    ):
        if tau is not None:
            tau = self.selector @ tau

        super().update(q, v, dv, tau)

        if tau is None:
            # TODO: we might have selector qfrc_u = B @ u
            # and we need to get u, it might be ambiguous
            ...

        # update numerical q, v, dv, tau
        self._q = self._back._q
        self._v = self._back._v
        self._dv = self._back._dv
        self._tau = self._back._tau

        # we have to compute dynamical quantities again
        self._inertia = None
        self._gravity = None
        self._coriolis = None
        self._bias_force = None
        self._momentum = None
        self._lagrangian = None
        self._contact_qforce = None
        self._coriolis_matrix = None

        # update each body
        for body in self.bodies.values():
            body.update()

    @property
    def inertia(self):
        return self._back.inertia_matrix

    @property
    def gravity(self):
        if self._gravity is None:
            self._gravity = self._back.rnea_fn(
                self._q,
                np.zeros_like(self._v),
                np.zeros_like(self._dv),
            )

        return self._gravity

    @property
    def coriolis(self):
        if self._coriolis is None:
            self._coriolis = self.bias_force - self.gravity

        return self._coriolis

    @property
    def bias_force(self):
        if self._bias_force is None:
            self._bias_force = self._back.rnea_fn(
                self._q,
                self._v,
                np.zeros_like(self._dv),
            )

        return self._bias_force

    @property
    def momentum(self):
        raise NotImplementedError

    @property
    def lagrangian(self):
        return self._back.kinetic_energy - self._back.potential_energy

    @property
    def contact_forces(self) -> List[ArrayLike]:
        if not self._contact_forces:
            self._contact_forces = []
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                self._contact_forces.append(body.contact._force)

        return self._contact_forces

    @property
    def contact_names(self):
        if not self._contact_names:
            self._contact_names = []
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                self._contact_names.append(body.name)

        return self._contact_names

    @property
    def contact_qforce(self):
        if not self._contact_qforce:
            qforce_sum = 0
            for body in self.bodies.values():
                if body.contact is None:
                    continue

                qforce_sum += body.contact.contact_qforce

    @property
    def coriolis_matrix(self):
        raise NotImplementedError

    @property
    def forward_dynamics(self):
        return self._dv

    @property
    def inverse_dynamics(self):
        return self._tau

    @property
    def state_space(self):
        raise NotImplementedError

    @property
    def selector(self):
        raise NotImplementedError

    def update_selector(self, matrix=None, passive_joints=None):
        super().update_selector(matrix, passive_joints)
