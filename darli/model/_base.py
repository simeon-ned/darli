from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
from ..utils.arrays import ArrayLike
from ..backend import BackendBase, Frame, CentroidalDynamics


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


class ContactBase(ABC):
    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def backend(self) -> BackendBase:
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def ref_frame(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @property
    @abstractmethod
    def jacobian(self):
        pass

    @property
    @abstractmethod
    def force(self):
        pass

    @force.setter
    @abstractmethod
    def force(self, value):
        pass

    @property
    @abstractmethod
    def qforce(self):
        pass

    @property
    @abstractmethod
    def cone(self):
        pass

    @abstractmethod
    def add_cone(self, mu, X=None, Y=None):
        pass


@dataclass
class FrameQuantity:
    local: ArrayLike
    world: ArrayLike
    world_aligned: ArrayLike


class BodyBase(ABC):
    @property
    @abstractmethod
    def backend(self):
        pass

    @property
    @abstractmethod
    def contact(self) -> ContactBase:
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
    def jacobian(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_jacobian(self, frame: Frame) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def jacobian_dt(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_jacobian_dt(self, frame: Frame) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def linear_velocity(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_linear_velocity(self, frame: Frame) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def angular_velocity(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_angular_velocity(self, frame: Frame) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def linear_acceleration(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_linear_acceleration(self, frame: Frame) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def angular_acceleration(self) -> FrameQuantity:
        pass

    @abstractmethod
    def get_angular_acceleration(self, frame: Frame) -> ArrayLike:
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
    def expression_model(self):
        """
        Property should return the underlying model with expressions instead of functions
        """
        pass

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
    def selector(self):
        pass

    @abstractmethod
    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        ...

    @abstractmethod
    def body(self, name: str) -> BodyBase:
        ...

    @abstractmethod
    def centroidal_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CentroidalDynamics:
        ...

    @abstractmethod
    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        ...
