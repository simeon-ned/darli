"""Module for base class of integrators."""

from abc import ABC, abstractmethod
from ...model._base import ModelBase
from ...utils.arrays import ArrayLike
import casadi as cs


class IntegratorBase(ABC):
    """
    Abstract base class for integrators.
    This class provides the interface that all derived integrator classes should implement.
    """

    def __init__(self, model: ModelBase):
        """
        Initialize the IntegratorBase with a model of the dynamic system.

        Args:
            model (ModelBase): The model on which the integration will be performed.
        """
        # simplify notation for integration on manifold

    @abstractmethod
    def tangent_step(self, x: ArrayLike, tang_x: ArrayLike) -> ArrayLike:
        """
        Abstract method to perform step assuming the configuration is on a manifold.

        Args:
            x (ArrayLike): The current state vector of the system.
            tang_x (ArrayLike): The tangent vector at state x.

        Returns:
            ArrayLike: The state vector after performing the tangent step.
        """

    @abstractmethod
    def forward(self, x0: ArrayLike, u: ArrayLike, dt: float | cs.SX) -> ArrayLike:
        """
        Abstract method to perform a forward integration step.

        Args:
             x0 (ArrayLike):
                The initial state from which to start the integration.
            u (ArrayLike): The input forces affecting the system's dynamics.
            dt (float | cs.SX):
                The timestep over which to integrate.

        Returns:
            ArrayLike: The state vector after performing the forward integration.
        """

    def derivative(self, x: ArrayLike, u: ArrayLike) -> ArrayLike:
        """Calculate the derivative (tangent) of the state vector for particular model"""


class Integrator(IntegratorBase):
    def __init__(self, model: ModelBase):
        """
        Initialize the Integrator with a dynamic system model.

        Args:
            model (ModelBase): The model on which the integration will be performed.
        """
        super().__init__(model)
        self.__model = model
        self.nq = self.__model.nq
        self.nv = self.__model.nv
        self.__manifold_integrate = self.__model.backend.integrate_configuration

    def tangent_step(self, x: ArrayLike, tang_x: ArrayLike) -> ArrayLike:
        """
        Concrete implementation of the tangent step for manifold integration.

        Args:
            x (ArrayLike): The current state vector of the system.
            tang_x (ArrayLike): The tangent vector at state x.

        Returns:
            ArrayLike: The state vector after performing the tangent step.
        """
        # Create a container for the new state
        container = self.__model.backend.math.zeros(self.nq + self.nv)
        # Integrate configuration on the manifold
        container[: self.nq] = self.__manifold_integrate(
            q=x[: self.nq], v=tang_x[: self.nv]
        )
        # Velocity and acceleration are simply updated
        container[self.nq :] = x[self.nq :] + tang_x[self.nv :]

        return container.array

    def derivative(self, x: ArrayLike, u: ArrayLike) -> ArrayLike:
        q, v = x[: self.nq], x[self.nq :]
        container = self.__model.backend.math.zeros(2 * self.nv)
        container[: self.nv] = v
        container[self.nv :] = self.__model.forward_dynamics(q, v, u)
        return container.array
