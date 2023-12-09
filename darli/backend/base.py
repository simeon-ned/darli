from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict
from ..arrays import ArrayLike, ArrayLikeFactory


class Frame(Enum):
    LOCAL = 1
    WORLD = 2
    LOCAL_WORLD_ALIGNED = 3

    @classmethod
    def from_str(cls, string: str) -> "Frame":
        if string == "local":
            return cls.LOCAL
        elif string == "world":
            return cls.WORLD
        elif string == "world_aligned":
            return cls.LOCAL_WORLD_ALIGNED
        else:
            raise ValueError(f"Unknown frame type: {string}")


@dataclass
class BodyInfo:
    position: ArrayLike
    rotation: ArrayLike
    jacobian: Dict[Frame, ArrayLike]
    djacobian: Dict[Frame, ArrayLike]
    lin_vel: Dict[Frame, ArrayLike]
    ang_vel: Dict[Frame, ArrayLike]
    lin_acc: Dict[Frame, ArrayLike]
    ang_acc: Dict[Frame, ArrayLike]


class BackendBase(ABC):
    math: ArrayLikeFactory

    @property
    @abstractmethod
    def nq(self) -> int:
        pass

    @property
    @abstractmethod
    def nv(self) -> int:
        pass

    @abstractmethod
    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def rnea(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def aba(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def potential_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        pass