from darli.backend import BackendBase, CasadiBackend, CentroidalDynamics
from darli.utils.arrays import ArrayLike
import casadi as cs

from .._base import Energy, CoM
from .._model import Model, ModelBase
from .._body import Body

# from ...model.model import Model, ModelBase
# from .state_space import FunctionalStateSpace
from ._body import FunctionalBody

from typing import List, Dict


class Functional(ModelBase):
    def __init__(self, backend: BackendBase):
        assert isinstance(
            backend, CasadiBackend
        ), "Symbolic Model only works with Casadi backend"
        self._backend = backend

        self.__model = Model(backend)

        # instances we want to cache
        self.__com = None
        self.__energy = None
        self.__centroidal = None

    @property
    def expression_model(self):
        return self.__model

    @property
    def q(self) -> ArrayLike:
        return self.__model.q

    @property
    def v(self) -> ArrayLike:
        return self.__model.v

    @property
    def dv(self) -> ArrayLike:
        return self.__model.dv

    @property
    def qfrc_u(self) -> ArrayLike:
        return self.__model.qfrc_u

    @property
    def backend(self) -> BackendBase:
        return self.__model.backend

    @property
    def nq(self) -> int:
        return self.__model.backend.nq

    @property
    def nv(self) -> int:
        return self.__model.backend.nv

    @property
    def nu(self) -> int:
        return self.__model.nu

    @property
    def nbodies(self) -> int:
        return self.__model.backend.nbodies

    @property
    def q_min(self) -> ArrayLike:
        return self.__model.backend.q_min

    @property
    def q_max(self) -> ArrayLike:
        return self.__model.backend.q_max

    @property
    def joint_names(self) -> List[str]:
        return self.__model.backend.joint_names

    @property
    def bodies(self) -> Dict[str, FunctionalBody]:
        # TODO: probably we should map each element to FunctionalBody too
        return self.__model.bodies

    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        return self.__model.add_body(bodies_names, Body)

    def body(self, name: str) -> FunctionalBody:
        return FunctionalBody.from_body(self.__model.body(name))

    # @property
    # def state_space(self):
    #     return FunctionalStateSpace.from_space(self.__model.state_space)

    @property
    def selector(self):
        return self.__model.selector

    def joint_id(self, name: str) -> int:
        return self.__model.joint_id(name)

    @property
    def contact_forces(self) -> List[ArrayLike]:
        return self.__model.contact_forces

    @property
    def contact_names(self) -> List[str]:
        return self.__model.contact_names

    def update_selector(
        self,
        matrix: ArrayLike | None = None,
        passive_joints: List[str | int] | None = None,
    ):
        self.__model.update_selector(matrix, passive_joints)

    @property
    def gravity(self) -> cs.Function:
        return cs.Function(
            "gravity",
            [self.q],
            [self.__model.gravity(self.q)],
            ["q"],
            ["gravity"],
        )

    @property
    def com(self) -> CoM:
        if self.__com is not None:
            return self.__com

        supercom = self.__model.com(self.q, self.v, self.dv)

        self.__com = CoM(
            position=cs.Function(
                "com_position",
                [self.q],
                [supercom.position],
                ["q"],
                ["com_position"],
            ),
            jacobian=cs.Function(
                "com_jacobian",
                [self.q],
                [supercom.jacobian],
                ["q"],
                ["com_jacobian"],
            ),
            velocity=cs.Function(
                "com_velocity",
                [self.q, self.v],
                [supercom.velocity],
                ["q", "v"],
                ["com_velocity"],
            ),
            acceleration=cs.Function(
                "com_acceleration",
                [self.q, self.v, self.dv],
                [supercom.acceleration],
                ["q", "v", "dv"],
                ["com_acceleration"],
            ),
            jacobian_dt=cs.Function(
                "com_jacobian_dt",
                [self.q, self.v],
                [supercom.jacobian_dt],
                ["q", "v"],
                ["com_jacobian_dt"],
            ),
        )
        return self.__com

    @property
    def energy(self) -> Energy:
        if self.__energy is not None:
            return self.__energy

        superenergy = self.__model.energy(self.q, self.v)

        self.__energy = Energy(
            kinetic=cs.Function(
                "kinetic_energy",
                [self.q, self.v],
                [superenergy.kinetic],
                ["q", "v"],
                ["kinetic_energy"],
            ),
            potential=cs.Function(
                "potential_energy",
                [self.q],
                [superenergy.potential],
                ["q"],
                ["potential_energy"],
            ),
        )
        return self.__energy

    @property
    def inertia(self) -> ArrayLike:
        return cs.Function(
            "inertia",
            [self.q],
            [self.__model.inertia(self.q)],
            ["q"],
            ["inertia"],
        )

    @property
    def coriolis(self) -> ArrayLike:
        return cs.Function(
            "coriolis",
            [self.q, self.v],
            [self.__model.coriolis(self.q, self.v)],
            ["q", "v"],
            ["coriolis"],
        )

    @property
    def bias_force(self) -> ArrayLike:
        return cs.Function(
            "bias_force",
            [self.q, self.v],
            [self.__model.bias_force(self.q, self.v)],
            ["q", "v"],
            ["bias_force"],
        )

    @property
    def momentum(self) -> ArrayLike:
        return cs.Function(
            "momentum",
            [self.q, self.v],
            [self.__model.momentum(self.q, self.v)],
            ["q", "v"],
            ["momentum"],
        )

    @property
    def lagrangian(self) -> ArrayLike:
        return cs.Function(
            "lagrangian",
            [self.q, self.v],
            [self.__model.lagrangian(self.q, self.v)],
            ["q", "v"],
            ["lagrangian"],
        )

    @property
    def contact_qforce(self) -> ArrayLike:
        return cs.Function(
            "contact_qforce",
            [self.q, *self.__model.contact_forces],
            [self.__model.contact_qforce],
            ["q", *self.contact_names],
            ["contact_qforce"],
        )

    @property
    def coriolis_matrix(self) -> ArrayLike:
        return cs.Function(
            "coriolis_matrix",
            [self.q, self.v],
            [self.__model.coriolis_matrix(self.q, self.v)],
            ["q", "v"],
            ["coriolis_matrix"],
        )

    @property
    def forward_dynamics(self) -> ArrayLike:
        return cs.Function(
            "forward_dynamics",
            [
                self.q,
                self.v,
                self.qfrc_u,
                *self.contact_forces,
            ],
            [self.__model.forward_dynamics(self.q, self.v, self.qfrc_u)],
            ["q", "v", "tau", *self.contact_names],
            ["dv"],
        )

    @property
    def inverse_dynamics(self) -> ArrayLike:
        return cs.Function(
            "inverse_dynamics",
            [
                self.q,
                self.v,
                self.dv,
                *self.__model.contact_forces,
            ],
            [self.__model.inverse_dynamics(self.q, self.v, self.dv)],
            ["q", "v", "dv", *self.contact_names],
            ["tau"],
        )

    @property
    def centroidal_dynamics(self) -> CentroidalDynamics:
        if self.__centroidal is not None:
            return self.__centroidal

        supercentroidal = self.__model.centroidal_dynamics(self.q, self.v, self.dv)

        res = CentroidalDynamics(
            matrix=cs.Function(
                "Ag",
                [self.q],
                [supercentroidal.matrix],
                ["q"],
                ["Ag"],
            ),
            linear=cs.Function(
                "h_lin",
                [self.q, self.v],
                [supercentroidal.linear],
                ["q", "v"],
                ["h_lin"],
            ),
            angular=cs.Function(
                "h_ang",
                [self.q, self.v],
                [supercentroidal.angular],
                ["q", "v"],
                ["h_ang"],
            ),
            linear_dt=cs.Function(
                "dh_lin",
                [self.q, self.v, self.dv],
                [supercentroidal.linear_dt],
                ["q", "v", "dv"],
                ["dh_lin"],
            ),
            angular_dt=cs.Function(
                "dh_ang",
                [self.q, self.v, self.dv],
                [supercentroidal.angular_dt],
                ["q", "v", "dv"],
                ["dh_ang"],
            ),
            matrix_dt=cs.Function(
                "dh_dq",
                [self.q, self.v, self.dv],
                [supercentroidal.matrix_dt],
                ["q", "v", "dv"],
                ["dh_dq"],
            ),
            dynamics_jacobian_q=cs.Function(
                "dhdot_dq",
                [self.q, self.v, self.dv],
                [supercentroidal.dynamics_jacobian_q],
                ["q", "v", "dv"],
                ["dhdot_dq"],
            ),
            dynamics_jacobian_v=cs.Function(
                "dhdot_dv",
                [self.q, self.v, self.dv],
                [supercentroidal.dynamics_jacobian_v],
                ["q", "v", "dv"],
                ["dhdot_dv"],
            ),
            dynamics_jacobian_dv=cs.Function(
                "dhdot_da",
                [self.q, self.v, self.dv],
                [supercentroidal.dynamics_jacobian_dv],
                ["q", "v", "dv"],
                ["dhdot_da"],
            ),
        )

        # cache result
        self.__centroidal = res

        return res

    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        # dummy implementation to satisfy base class
        return
