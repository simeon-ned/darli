import numpy as np
from typing import List
import pinocchio as pin
from typing import Dict
from dataclasses import dataclass
import casadi as cs
from darli.models.body import Body
import numpy.typing as npt


@dataclass
class CoM:
    position: cs.Function | float
    velocity: cs.Function | float
    acceleration: cs.Function | float
    jacobian: cs.Function | float
    jacobian_dt: cs.Function | float


@dataclass
class Energy:
    kinetic: cs.Function | float
    potential: cs.Function | float


class RobotModel:
    def __init__(self, urdf_path: str, backend):
        self._back = backend

        self.__model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.__data: pin.Data = self.__model.createData()

        self.__joint_names = list(self.__model.names)
        self.__joint_names.remove("universe")

        self.__joint_map: Dict[str, int] = dict()
        for joint_name in self.__joint_names:
            self.__joint_map[joint_name] = self.joint_idx(joint_name)

        self.__bodies: Dict[str, Body] = dict()

        self._forward_dynamics = None
        self._inverse_dynamics = None
        self._gravity = None
        self._coriolis = None
        self._bias_force = None
        self._momentum = None
        self._lagrangian = None
        self._contact_qforce = None
        self._coriolis_matrix = None

        self._contact_forces = None
        self._contact_names = None

        self.__com: CoM | None = None
        self.__energy: Energy | None = None

        # number of actuators might be adjusted with a change of selector
        self.__nu: int = self.nv
        self.__selector: npt.ArrayLike = None
        self.update_selector()

    def joint_idx(self, joint_name: str) -> int:
        return self._back.joint_iq(joint_name)

    def update(self, q, v, dv=None, tau=None):
        self._back.update(q, v, dv, tau)

    @property
    def bodies(self) -> Dict[str, Body]:
        return self.__bodies

    def body(self, body_name) -> Body:
        try:
            return self.bodies[body_name]
        except KeyError:
            raise KeyError(f"Body {body_name} is not added")

    def add_bodies(self, bodies_names: List[str] | Dict[str, str]):
        # """Adds bodies to the model and update the model"""
        if not bodies_names or len(bodies_names) == 0:
            return

        if isinstance(bodies_names, dict):
            self.bodies_names = bodies_names
            for body_pairs in self.bodies_names.items():
                body = Body(name=dict([body_pairs]), kindyn_backend=self._back)
                self.bodies[body_pairs[0]] = body
        elif isinstance(bodies_names, list):
            self.bodies_names = set(bodies_names)
            for body_name in self.bodies_names:
                body = Body(name=body_name, kindyn_backend=self._back)
                self.bodies[body_name] = body
        else:
            raise TypeError(
                f"unknown type of mapping is passed to add bodies: {type(bodies_names)}"
            )

        # update what we have to compute again due to changes
        self._contact_forces = None
        self._contact_names = None
        self._contact_qforce = None

        # self.state_space.update()

    @property
    def com(self) -> CoM:
        if not self.__com:
            self.__com = CoM(
                position=self._back.com["position"],
                velocity=self._back.com["velocity"],
                acceleration=self._back.com["acceleration"],
                jacobian=self._back.com["jacobian"],
                jacobian_dt=self._back.com["jacobian_dt"],
            )

        return self.__com

    @property
    def energy(self) -> Energy:
        if not self.__energy:
            self.__energy = Energy(
                kinetic=self._back.kinetic_energy,
                potential=self._back.potential_energy,
            )

        return self.__energy

    @property
    def joint_map(self) -> Dict[str, int]:
        return self.__joint_map

    @property
    def q_min(self) -> np.ndarray:
        return self.__model.lowerPositionLimit

    @property
    def q_max(self) -> np.ndarray:
        return self.__model.upperPositionLimit

    @property
    def nq(self) -> int:
        return self.__model.nq

    @property
    def nv(self) -> int:
        return self.__model.nv

    @property
    def nu(self) -> int:
        return self.__nu
        # return self.nv

    @nu.setter
    def nu(self, value: int):
        self.__nu = value

    @property
    def mass(self) -> float:
        return pin.computeTotalMass(self.__model, self.__data)

    @property
    def joint_names(self) -> List[str]:
        return self.__joint_names

    @property
    def inertia(self):
        raise NotImplementedError

    @property
    def gravity(self):
        raise NotImplementedError

    @property
    def coriolis(self):
        raise NotImplementedError

    @property
    def bias_force(self):
        raise NotImplementedError

    @property
    def momentum(self):
        raise NotImplementedError

    @property
    def lagrangian(self):
        raise NotImplementedError

    @property
    def contact_forces(self):
        raise NotImplementedError

    @property
    def contact_names(self):
        raise NotImplementedError

    @property
    def contact_qforce(self):
        raise NotImplementedError

    @property
    def coriolis_matrix(self):
        raise NotImplementedError

    @property
    def forward_dynamics(self):
        raise NotImplementedError

    @property
    def inverse_dynamics(self):
        raise NotImplementedError

    @property
    def state_space(self):
        raise NotImplementedError

    @property
    def selector(self):
        return self.__selector

    def update_selector(
        self,
        matrix: npt.ArrayLike | None = None,
        passive_joints: List[str | int] | None = None,
    ):
        self.__selector = np.eye(self.nv)

        if matrix is not None:
            self.__selector = matrix
            self.nu = np.shape(self._selector)[1]

        if passive_joints is not None and matrix is None:
            self.nu = self.nv - len(passive_joints)
            joint_id = []
            for joint in passive_joints:
                if isinstance(joint, str):
                    joint_id.append(self.joint_idx(joint))
                if isinstance(joint, int):
                    joint_id.append(joint)
            self.__selector = np.delete(self.__selector, joint_id, axis=1)
