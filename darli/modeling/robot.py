from typing import List, Dict

from darli.backend import BackendBase, CasadiBackend
from darli.arrays import ArrayLike
import numpy as np

from .base import Energy, CoM, ModelBase, BodyBase
from .body import Body
from .state_space import CasadiStateSpace, PinocchioStateSpace


class Robot(ModelBase):
    def __init__(self, backend: BackendBase):
        self._backend = backend

        self._q = self._backend._q
        self._v = self._backend._v
        self._dv = self._backend._dv

        # force applied in joint space
        self._qfrc_u = self._backend._tau

        # force applied to actuators, i.e. selector @ qfrc_u
        self._u = self._backend._tau

        self.__bodies: Dict[str, BodyBase] = dict()
        self.update_selector()

        if isinstance(self.backend, CasadiBackend):
            self.__state_space = CasadiStateSpace(self)
        else:
            self.__state_space = PinocchioStateSpace(self)

    @property
    def q(self) -> ArrayLike:
        return self._q

    @property
    def v(self) -> ArrayLike:
        return self._v

    @property
    def dv(self) -> ArrayLike:
        return self._dv

    @property
    def qfrc_u(self) -> ArrayLike:
        return self._qfrc_u

    @property
    def backend(self) -> BackendBase:
        return self._backend

    @property
    def nq(self) -> int:
        return self._backend.nq

    @property
    def nv(self) -> int:
        return self._backend.nv

    @property
    def nu(self) -> int:
        return np.shape(self.__selector)[1]

    @property
    def nbodies(self) -> int:
        return self._backend.nbodies

    @property
    def q_min(self) -> ArrayLike:
        return self._backend.q_min

    @property
    def q_max(self) -> ArrayLike:
        return self._backend.q_max

    @property
    def joint_names(self) -> List[str]:
        return self._backend.joint_names

    @property
    def bodies(self) -> Dict[str, BodyBase]:
        return self.__bodies

    def add_body(self, bodies_names: List[str] | Dict[str, str], constructor=Body):
        if not bodies_names or len(bodies_names) == 0:
            return

        if isinstance(bodies_names, dict):
            for body_pairs in bodies_names.items():
                body = constructor(name=dict([body_pairs]), backend=self._backend)
                self.__bodies[body_pairs[0]] = body
        elif isinstance(bodies_names, list):
            for body_name in bodies_names:
                body = constructor(name=body_name, backend=self._backend)
                self.__bodies[body_name] = body
        else:
            raise TypeError(
                f"unknown type of mapping is passed to add bodies: {type(bodies_names)}"
            )

    def body(self, name: str) -> BodyBase:
        assert name in self.__bodies, f"Body {name} is not added"

        return self.__bodies[name]

    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        self._q = q
        self._v = v

        # if we pass u, we assume it is already in actuator space
        self._backend.update(q, v, dv, self.selector @ u if u is not None else None)

        # update current dv and qfrc_u
        self._dv = self._backend._dv
        self._qfrc_u = self._backend._tau
        if u:
            self._u = u

        # update bodies
        for body in self.__bodies.values():
            body.update()

    def gravity(self, q: ArrayLike | None = None) -> ArrayLike:
        return self._backend.rnea(
            q if q is not None else self._q,
            self._backend.math.zeros(self._backend.nv).array,
            self._backend.math.zeros(self._backend.nv).array,
        )

    def inertia(self, q: ArrayLike | None = None) -> ArrayLike:
        return self._backend.inertia_matrix(q if q is not None else self._q)

    def com(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CoM:
        return CoM(
            position=self._backend.com_pos(
                q if q is not None else self._q,
            ),
            velocity=self._backend.com_vel(
                q if q is not None else self._q,
                v if v is not None else self._v,
            ),
            acceleration=self._backend.com_acc(
                q if q is not None else self._q,
                v if v is not None else self._v,
                dv if dv is not None else self._dv,
            ),
            jacobian=self._backend.jacobian(
                q if q is not None else self._q,
            ),
            jacobian_dt=self._backend.jacobian_dt(
                q if q is not None else self._q,
                v if v is not None else self._v,
            ),
        )

    def energy(self, q: ArrayLike | None = None, v: ArrayLike | None = None) -> Energy:
        return Energy(
            kinetic=self._backend.kinetic_energy(
                q if q is not None else self._q,
                v if v is not None else self._v,
            ),
            potential=self._backend.potential_energy(
                q if q is not None else self._q,
            ),
        )

    def coriolis(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        tau_grav = self._backend.rnea(
            q if q is not None else self._q,
            self._backend.math.zeros(self._backend.nv).array,
            self._backend.math.zeros(self._backend.nv).array,
        )

        tau_bias = self._backend.rnea(
            q if q is not None else self._q,
            v if v is not None else self._v,
            self._backend.math.zeros(self._backend.nv).array,
        )

        return tau_grav - tau_bias

    def bias_force(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self._backend.rnea(
            q if q is not None else self._q,
            v if v is not None else self._v,
            self._backend.math.zeros(self._backend.nv).array,
        )

    def momentum(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        raise NotImplementedError("Implement me pls")  # FIXME: implement me

    def lagrangian(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self._backend.kinetic_energy(
            q if q is not None else self._q,
            v if v is not None else self._v,
        ) - self._backend.potential_energy(
            q if q is not None else self._q,
        )

    @property
    def contact_forces(self) -> List[ArrayLike]:
        forces = []
        for body in self.__bodies.values():
            if body.contact is not None:
                forces.append(body.contact.force)

        return forces

    @property
    def contact_names(self) -> List[str]:
        names = []
        for body in self.__bodies.values():
            if body.contact is not None:
                names.append(body.contact.name)

        return names

    @property
    def contact_qforce(self) -> ArrayLike:
        qforce = self.backend.math.zeros(self.nv).array
        for body in self.__bodies.values():
            if body.contact is not None:
                qforce += body.contact.qforce

        return qforce

    def coriolis_matrix(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        raise NotImplementedError(
            "Move me to backend, because it can be easily done only for casadi"
        )

    def forward_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        # if u is not passed, we assume and take current tau, otherwise premultiply by selector matrix
        return self._backend.aba(
            q if q is not None else self._q,
            v if v is not None else self._v,
            tau=(self._qfrc_u if u is None else self.selector @ u)
            + self.contact_qforce,
        )

    def inverse_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Returns qfrc_u
        """
        return (
            self._backend.rnea(
                q if q is not None else self._q,
                v if v is not None else self._v,
                dv if dv is not None else self._dv,
            )
            - self.contact_qforce
        )

    @property
    def state_space(self):
        return self.__state_space

    @property
    def selector(self):
        return self.__selector

    def joint_id(self, name: str) -> int:
        return self._backend.joint_id(name)

    def update_selector(
        self,
        matrix: ArrayLike | None = None,
        passive_joints: List[str | int] | None = None,
    ):
        self.__selector = np.eye(self.nv)

        if matrix is not None:
            self.__selector = matrix

        if passive_joints is not None:
            joint_id = []
            for joint in passive_joints:
                if isinstance(joint, str):
                    joint_id.append(self._backend.joint_id(joint))
                elif isinstance(joint, int):
                    joint_id.append(joint)
                else:
                    raise TypeError(
                        f"unknown type of joint is passed to add bodies: {type(joint)}"
                    )

            self.__selector = np.delete(self.__selector, joint_id, axis=1)

        # update qfrc_u
        if isinstance(self.backend, CasadiBackend):
            self._qfrc_u = self.backend.math.array("tau", self.nu).array
        else:
            self._qfrc_u = self.backend.math.zeros(self.nu).array
