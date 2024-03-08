"""Module for forward euler integrator."""

from ._base import Integrator, ModelBase, ArrayLike, cs


class ForwardEuler(Integrator):
    """
    Implements the Euler method for numerical integration.

    The Euler method is a simple first-order integration method that updates
    the state by taking a single step forward using the derivative of the
    state at the current position. This method can be used for systems
    evolving on manifolds and is computationally less intensive than higher-order methods.
    """

    # def __init__(self, model: ModelBase):
    #     """
    #     Initialize the Euler integrator with a model that defines system dynamics,
    #     which may include evolution on a manifold.

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
        Perform a single Euler integration step.

        Args:
            derivative: A function that computes the tangent of the state.
            x0: Initial state vector, which may include manifold-valued components.
            u: Inputs forces acting on the system.
            dt: Time step for integration.

        Returns:
            The estimated state vector after the Euler integration step.
        """

        # Calculate the derivative at the current position
        log = self.derivative(x0, u)

        # Update the state on manifold by taking a step along the tangent
        # and projecting back onto the manifold
        exp = self.tangent_step(x0, dt * log)

        return exp
