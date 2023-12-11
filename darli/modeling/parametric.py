from typing import List, Dict

from darli.backend import BackendBase
from darli.arrays import ArrayLike
from .body import Body
from .base import Energy, CoM, ModelBase


class ParametricRobot(ModelBase):
    def __init__(self, backend: BackendBase):

        self._backend = backend

        self._q = self._backend.math.zeros(self.nq).array
        self._v = self._backend.math.zeros(self.nv).array
        self._dv = self._backend.math.zeros(self.nv).array
        self._tau = self._backend.math.zeros(self.nv).array

        self._parameters = self._backend.math.zeros(self.nbodies * 10).array
        self.__bodies: Dict[str, Body] = dict()

    @property
    def nq(self) -> int:
        return self._backend.nq

    @property
    def nv(self) -> int:
        return self._backend.nv

    @property
    def nbodies(self) -> int:
        return self._backend.nbodies

    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        if not bodies_names or len(bodies_names) == 0:
            return

        if isinstance(bodies_names, dict):
            for body_pairs in bodies_names.items():
                body = Body(name=dict([body_pairs]), backend=self._backend)
                self.__bodies[body_pairs[0]] = body
        elif isinstance(bodies_names, list):
            for body_name in bodies_names:
                body = Body(name=body_name, backend=self._backend)
                self.__bodies[body_name] = body
        else:
            raise TypeError(
                f"unknown type of mapping is passed to add bodies: {type(bodies_names)}"
            )

    def body(self, name: str) -> Body:
        assert name in self.__bodies, f"Body {name} is not added"

        return self.__bodies[name]

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
