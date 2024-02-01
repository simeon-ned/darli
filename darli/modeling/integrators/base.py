"""Module for base class of integrators"""

from abc import ABC, abstractstaticmethod, abstractmethod
from ..base import StateSpaceBase, ModelBase
from ...arrays import ArrayLike
import casadi as cs


class IntegratorBase(ABC):
    def __init__(self, model: ModelBase):
        """Base class for integrators."""
        # simplify notation for integration on manifold

    @abstractmethod
    def tangent_step(self, x: ArrayLike, tang_x: ArrayLike) -> ArrayLike:
        """
        Perform integration step assuming configuration on manifold.
        """

    @abstractmethod
    def forward(
        derivative: ArrayLike,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        dt: float | cs.SX,
    ) -> ArrayLike:
        """
        Perform integration step assuming configuration on manifold.
        """


class Integrator(IntegratorBase):
    def __init__(self, model: ModelBase):
        super().__init__(model)

        self.nq = model.nq
        self.nv = model.nv
        self.__manifold_int = model.backend.integrate_configuration
        self.__container = model.backend.math.zeros(self.nq + self.nv)

    def tangent_step(self, x: ArrayLike, tang_x: ArrayLike) -> ArrayLike:
        # configuration is integrated on manifold using pinocchio implementation
        self.__container[: self.nq] = self.__manifold_int(
            q=x[: self.nq], v=tang_x[: self.nv]
        )

        # velocity and acceleration are in the same space and do not require specific treatment
        self.__container[self.nq :] = x[self.nq :] + tang_x[self.nv :]

        return self.__container.array
