from darli.backend import BackendBase, CasadiBackend
from darli.arrays import ArrayLike
import casadi as cs


class Robot:
    def __init__(self, backend: BackendBase):
        self._backend = backend

    def gravity(self, q: ArrayLike | None = None) -> ArrayLike:
        return self._backend.rnea(
            q,
            self._backend.array_factory.zeros(self._backend.nv).array,
            self._backend.array_factory.zeros(self._backend.nv).array,
        )


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
                    self._backend.array_factory.zeros(self._backend.nv).array,
                    self._backend.array_factory.zeros(self._backend.nv).array,
                )
            ],
            ["q"],
            ["tau"],
        )
