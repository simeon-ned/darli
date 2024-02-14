"""Module for forward euler integrator"""
from ._base import Integrator, ModelBase, ArrayLike, cs


class MidPoint(Integrator):
    """
    Implements the Midpoint method for numerical integration.

    The Midpoint method is a second-order method that computes the derivative
    at the midpoint of the interval to update the state. It provides a better
    approximation than the Euler method with modest computational requirements.
    This method can handle systems with state spaces that include manifolds.
    """

    # def __init__(self, model: ModelBase):
    #     """
    #     Initialize the Midpoint integrator with a model that defines system dynamics,
    #     which may include manifold evolution.

    #     Args:
    #         model (ModelBase): The DARLI model providing the system dynamics.
    #     """
    #     super().__init__(model)

    def forward(
        self,
        x0: ArrayLike,
        u: ArrayLike,
        dt: cs.SX | float,
    ) -> ArrayLike:
        """
        Perform a single Midpoint integration step.

        Args:
            derivative: A function that computes the tangent of the state.
            x0: Initial state vector, which may include manifold-valued components.
            u: Inputs forces acting on the system.
            dt: Time step for integration.

        Returns:
            The estimated state vector after the Midpoint integration step.
        """
        # Compute the first time derivative of the state (Euler step)
        k1_log = self.derivative(x0, u)

        # Compute the state at the mid-point using the initial slope (k1_log)
        k2_exp = self.tangent_step(x0, 0.5 * dt * k1_log)

        # Compute the time derivative at the mid-point, based on the estimated state (k2_exp)
        k2_log = self.derivative(k2_exp, u)

        # Perform the full step using the slope at the mid-point (k2_log)
        return self.tangent_step(x0, dt * k2_log)
