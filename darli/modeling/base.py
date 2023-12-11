from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..arrays import ArrayLike
from typing import List, Dict
from .body import Body
from ..backend import PinocchioBased


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
