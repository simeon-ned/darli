from abc import ABC, abstractmethod
from ..utils.arrays import ArrayLike
from ..model._base import ModelBase


class StateSpaceBase(ABC):
    @property
    def integrator(self):
        pass

    @property
    @abstractmethod
    def model(self) -> ModelBase:
        pass

    @property
    @abstractmethod
    def force_jacobians(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def time_variation(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ):
        pass

    @abstractmethod
    def derivative(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ):
        pass

    @property
    @abstractmethod
    def state_jacobian(self):
        pass

    @property
    @abstractmethod
    def input_jacobian(self):
        pass

    @abstractmethod
    def force_jacobian(self, body_name: str):
        pass
