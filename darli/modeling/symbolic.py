from darli.backend import BackendBase, CasadiBackend
from darli.arrays import ArrayLike
import casadi as cs

from .base import Energy, CoM
from .robot import Robot


class Symbolic(Robot):
    def __init__(self, backend: BackendBase):
        super().__init__(backend)

        assert isinstance(
            backend, CasadiBackend
        ), "Symbolic robot only works with Casadi backend"
        self._backend = backend

    @property
    def gravity(self) -> cs.Function:
        return cs.Function(
            "gravity",
            [self._backend._q],
            [super().gravity()],
            ["q"],
            ["gravity"],
        )

    def com(self) -> CoM:
        supercom = super().com()

        return CoM(
            position=cs.Function(
                "com_position",
                [self._backend._q],
                [supercom.position],
                ["q"],
                ["com_position"],
            ),
            jacobian=cs.Function(
                "com_jacobian",
                [self._backend._q],
                [supercom.jacobian],
                ["q"],
                ["com_jacobian"],
            ),
            velocity=cs.Function(
                "com_velocity",
                [self._backend._q, self._backend._v],
                [supercom.velocity],
                ["q", "v"],
                ["com_velocity"],
            ),
            acceleration=cs.Function(
                "com_acceleration",
                [self._backend._q, self._backend._v, self._backend._dv],
                [supercom.acceleration],
                ["q", "v", "dv"],
                ["com_acceleration"],
            ),
            jacobian_dt=cs.Function(
                "com_jacobian_dt",
                [self._backend._q, self._backend._v],
                [supercom.jacobian_dt],
                ["q", "v"],
                ["com_jacobian_dt"],
            ),
        )

    def energy(self, q: ArrayLike | None = None, v: ArrayLike | None = None) -> Energy:
        superenergy = super().energy(q, v)

        return Energy(
            kinetic=cs.Function(
                "kinetic_energy",
                [self._backend._q, self._backend._v],
                [superenergy.kinetic],
                ["q", "v"],
                ["kinetic_energy"],
            ),
            potential=cs.Function(
                "potential_energy",
                [self._backend._q],
                [superenergy.potential],
                ["q"],
                ["potential_energy"],
            ),
        )

    def inertia(self) -> ArrayLike:
        return cs.Function(
            "inertia",
            [self._backend._q],
            [super().inertia()],
            ["q"],
            ["inertia"],
        )

    def coriolis(self) -> ArrayLike:
        return cs.Function(
            "coriolis",
            [self._backend._q, self._backend._v],
            [super().coriolis()],
            ["q", "v"],
            ["coriolis"],
        )

    def bias_force(self) -> ArrayLike:
        return cs.Function(
            "bias_force",
            [self._backend._q, self._backend._v],
            [super().bias_force()],
            ["q", "v"],
            ["bias_force"],
        )

    def momentum(self) -> ArrayLike:
        return cs.Function(
            "momentum",
            [self._backend._q, self._backend._v],
            [super().momentum()],
            ["q", "v"],
            ["momentum"],
        )

    def lagrangian(self) -> ArrayLike:
        return cs.Function(
            "lagrangian",
            [self._backend._q, self._backend._v],
            [super().lagrangian()],
            ["q", "v"],
            ["lagrangian"],
        )

    @property
    def contact_qforce(self) -> ArrayLike:
        return cs.Function(
            "contact_qforce",
            [self._backend._q, *self.contact_forces],
            [super().contact_qforce()],
            ["q", *self.contact_names],
            ["contact_qforce"],
        )

    def coriolis_matrix(self) -> ArrayLike:
        return cs.Function(
            "coriolis_matrix",
            [self._backend._q, self._backend._v],
            [super().coriolis_matrix()],
            ["q", "v"],
            ["coriolis_matrix"],
        )

    def forward_dynamics(self) -> ArrayLike:
        return cs.Function(
            "forward_dynamics",
            [
                self._backend._q,
                self._backend._v,
                self._backend._tau,
                *self.contact_forces,
            ],
            [super().forward_dynamics()],
            ["q", "v", "tau", *self.contact_names],
            ["dv"],
        )

    def inverse_dynamics(self) -> ArrayLike:
        return cs.Function(
            "inverse_dynamics",
            [
                self._backend._q,
                self._backend._v,
                self._backend._dv,
                *self.contact_forces,
            ],
            [super().inverse_dynamics()],
            ["q", "v", "dv", *self.contact_names],
            ["tau"],
        )
