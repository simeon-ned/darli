"""Module for RK4 integrator"""
from ._base import Integrator, ModelBase, ArrayLike, cs


class RK4(Integrator):
    """
    Implements the Runge-Kutta 4th order (RK4) method.

    This implementation of RK4 is capable of handling systems that evolve not just in
    Euclidean space but also on manifolds. This is particularly useful for
    models that include rotational dynamics, where the state variables (such as
    quaternions) evolve on a manifold.
    """

    # def __init__(self, model: ModelBase):
    #     """
    #     Initialize the RK4 integrator with a model that defines system dynamics,
    #     possibly on a manifold (quaternions in floating base).

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
        Perform a single RK4 integration step, suitable for state spaces that
        might include manifolds (floating base).

        Args:
            derivative: A function that computes the tangent of the state.
            x0: Initial state vector, which may include manifold-valued components.
            u: Inputs forces acting on the system.
            dt: Time step for integration.

        Returns:
            The estimated state vector after the RK4 integration step.
        """

        # Calculate the four increments from the derivative function
        k1_log = self.derivative(x0, u)  # perform log map
        k2_exp = self.tangent_step(x0, 0.5 * dt * k1_log)  # perform exponential map
        k2_log = self.derivative(k2_exp, u)
        k3_exp = self.tangent_step(x0, 0.5 * dt * k2_log)
        k3_log = self.derivative(k3_exp, u)
        k4_exp = self.tangent_step(x0, dt * k3_log)
        k4_log = self.derivative(k4_exp, u)

        # Combine the four increments for the final state estimate
        return self.tangent_step(
            x0, (dt / 6.0) * (k1_log + 2 * k2_log + 2 * k3_log + k4_log)
        )
