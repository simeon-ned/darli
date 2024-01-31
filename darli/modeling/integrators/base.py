"""Module for base class of integrators"""

from abc import ABC, abstractstaticmethod
from ..base import StateSpaceBase
from ...arrays import ArrayLike
import casadi as cs


class Integrator(ABC):
    @abstractstaticmethod
    def forward(
        state_space: StateSpaceBase,
        x0: ArrayLike,
        qfrc_u: ArrayLike,
        dt: float | cs.SX,
    ) -> ArrayLike:
        pass
