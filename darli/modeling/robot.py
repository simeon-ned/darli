from darli.backend import BackendBase, CasadiBackend
from darli.arrays import ArrayLike
import casadi as cs
from .base import PinocchioBased, Energy, CoM


class Robot(PinocchioBased):
    def __init__(self, backend: BackendBase, urdf_path: str):
        super().__init__(urdf_path)
        self._backend = backend

        self._q = self._backend.math.zeros(self._backend.nq).array
        self._v = self._backend.math.zeros(self._backend.nv).array
        self._dv = self._backend.math.zeros(self._backend.nv).array
        self._tau = self._backend.math.zeros(self._backend.nv).array

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

    def gravity(self, q: ArrayLike | None = None) -> ArrayLike:
        return self._backend.rnea(
            q,
            self._backend.math.zeros(self._backend.nv).array,
            self._backend.math.zeros(self._backend.nv).array,
        )

    def inertia(self, q: ArrayLike | None = None) -> ArrayLike:
        return self._backend.inertia_matrix(q if q else self._q)

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

    def inverse_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @property
    def state_space(self):
        pass

    @property
    def selector(self):
        pass


class Symbolic:
    def __init__(self, backend: BackendBase):
        assert isinstance(
            backend, CasadiBackend
        ), "Symbolic robot only works with Casadi backend"
        self._backend = backend

    def gravity(self) -> cs.Function:
        return cs.Function(
            "gravity",
            [self._backend._q],
            [
                self._backend.rnea(
                    self._backend._q,
                    self._backend.math.zeros(self._backend.nv).array,
                    self._backend.math.zeros(self._backend.nv).array,
                )
            ],
            ["q"],
            ["tau"],
        )
