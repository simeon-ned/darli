"""
This submodule provides implementation of various numerical integrators that are
designed to respect the manifold structure of configuration spaces, such as the
space of quaternions for representing orientations in 3D.

The key concept in these integrators is the use of tangent space (logarithm map)
for computations during each integration step. The results are then mapped back onto the
manifold using the exponential map, ensuring that the updated configurations remain
within the manifold.

Key References:
- Celledoni, E., & Owren, B. (2003). Lie group methods for rigid body dynamics and time
  integration on manifolds. Computer Methods in Applied Mechanics and Engineering, 192(3-4), 421-438.
  Available: https://www.sciencedirect.com/science/article/abs/pii/S0045782502005200

- Celledoni, E., Çokaj, E., Leone, A., Murari, D., & Owren, B. (2022). Lie group integrators for mechanical systems.
  International Journal of Computer Mathematics, 99(1), 58-88.
  Available: https://arxiv.org/pdf/2102.12778.pdf

Available Integrators:
- ForwardEuler: The forward Euler integration scheme.
- MidPoint: The mid-point integration scheme.
- RK4: The classical fourth-order Runge-Kutta method.
"""

from ._base import Integrator
from ._fwd_euler import ForwardEuler
from ._rk4 import RK4
from ._mid_point import MidPoint
