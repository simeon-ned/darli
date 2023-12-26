from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..arrays import ArrayLike
from typing import List, Dict
from ..backend import BackendBase, Frame


@dataclass
class CoM:
    position: ArrayLike
    velocity: ArrayLike
    acceleration: ArrayLike
    jacobian: ArrayLike
    jacobian_dt: ArrayLike


@dataclass
class Energy:
    kinetic: ArrayLike
    potential: ArrayLike


class BodyBase(ABC):
    @property
    @abstractmethod
    def backend(self):
        pass

    @property
    @abstractmethod
    def contact(self):
        pass

    @property
    @abstractmethod
    def position(self):
        pass

    @property
    @abstractmethod
    def rotation(self):
        pass

    @property
    @abstractmethod
    def quaternion(self):
        pass

    @property
    @abstractmethod
    def jacobian(self):
        pass

    @property
    @abstractmethod
    def jacobian_dt(self):
        pass

    @property
    @abstractmethod
    def linear_velocity(self):
        pass

    @property
    @abstractmethod
    def angular_velocity(self):
        pass

    @property
    @abstractmethod
    def linear_acceleration(self):
        pass

    @property
    @abstractmethod
    def angular_acceleration(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def add_contact(self, contact_type="point", frame=Frame.LOCAL_WORLD_ALIGNED):
        pass


class ModelBase(ABC):
    @property
    @abstractmethod
    def q(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def v(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def dv(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def qfrc_u(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def backend(self) -> BackendBase:
        pass

    @property
    @abstractmethod
    def bodies(self) -> Dict[str, BodyBase]:
        pass

    @property
    @abstractmethod
    def nq(self) -> int:
        pass

    @property
    @abstractmethod
    def nv(self) -> int:
        pass

    @property
    @abstractmethod
    def nu(self) -> int:
        pass

    @property
    @abstractmethod
    def q_min(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def q_max(self) -> ArrayLike:
        pass

    @abstractmethod
    def joint_id(self, name: str) -> int:
        pass

    @property
    @abstractmethod
    def nbodies(self) -> int:
        pass

    @abstractmethod
    def com(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CoM:
        pass

    @abstractmethod
    def energy(self, q: ArrayLike | None = None, v: ArrayLike | None = None) -> Energy:
        pass

    @abstractmethod
    def inertia(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def gravity(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def coriolis(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def bias_force(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def momentum(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def lagrangian(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def contact_forces(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def contact_names(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def contact_qforce(self) -> ArrayLike:
        pass

    @abstractmethod
    def coriolis_matrix(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def forward_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def inverse_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def state_space(self):
        pass

    @property
    @abstractmethod
    def selector(self):
        pass

    @abstractmethod
    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        ...

    @abstractmethod
    def body(self, name: str) -> BodyBase:
        ...
