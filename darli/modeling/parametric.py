from typing import List, Dict

from darli.backend import BackendBase
from darli.arrays import ArrayLike
from . import Body
from .base import PinocchioBased, Energy, CoM


class ParametricRobot(PinocchioBased):
    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        pass

    def body(self, name: str) -> Body:
        pass

    def __init__(
        self, backend: BackendBase, urdf_path: str
    ):  # TODO: we need urdf in all cases
        super().__init__(urdf_path)

        self._backend = backend

        self._q = self._backend.math.zeros(self.nq).array
        self._v = self._backend.math.zeros(self.nv).array
        self._dv = self._backend.math.zeros(self.nv).array
        self._tau = self._backend.math.zeros(self.nv).array

        self._parameters = self._backend.math.zeros(self.nbodies * 10).array

    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        self._q = q
        self._v = v
        self._dv = dv
        self._tau = tau
        return self._backend.update(q, v, dv, tau)

    def inverse_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        return (
            self._backend.torque_regressor(
                q if q is not None else self._q,
                v if v is not None else self._v,
                tau if tau is not None else self._tau,
            )
            @ self._parameters
        )

    def gravity(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.inverse_dynamics(
            q if q is not None else self._q,
            self._backend.math.zeros(self._backend.nv).array,
            self._backend.math.zeros(self._backend.nv).array,
        )

    def inertia(self, q: ArrayLike | None = None) -> ArrayLike:
        inertia = self._backend.math.zeros((self.nv, self.nv)).array
        unit_vectors = self._backend.math.eye(self.nv).array

        for i in range(self.nv):
            unit_vector = unit_vectors[i, :]
            inertia[:, i] = (
                self._backend.torque_regressor(
                    self._q, self._backend.math.zeros(self.nv).array, unit_vector
                )
                @ self._parameters
            )

        return inertia

    def com(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CoM:
        pass

    def energy(self, q: ArrayLike | None = None, v: ArrayLike | None = None) -> Energy:
        pass

    def coriolis(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def bias_force(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def momentum(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def lagrangian(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @property
    def contact_forces(self) -> ArrayLike:
        pass

    @property
    def contact_names(self) -> ArrayLike:
        pass

    @property
    def contact_qforce(self) -> ArrayLike:
        pass

    def coriolis_matrix(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def forward_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @property
    def state_space(self):
        pass

    @property
    def selector(self):
        pass
