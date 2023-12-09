from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..arrays import ArrayLike
import pinocchio as pin
from typing import List, Dict
from .body import Body, FrameQuantity


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


class ModelBase(ABC):
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
    def body(self, name: str) -> Body:
        ...


class PinocchioBased(ModelBase, ABC):
    def __init__(self, urdf_path: str) -> None:
        self._pinmodel: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self._pindata: pin.Data = self._pinmodel.createData()

    @property
    def nq(self) -> int:
        return self._pinmodel.nq

    @property
    def nv(self) -> int:
        return self._pinmodel.nv

    @property
    def nbodies(self) -> int:
        return self._pinmodel.nbodies - 1
