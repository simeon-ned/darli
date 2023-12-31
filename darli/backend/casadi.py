import casadi_kin_dyn.casadi_kin_dyn as ckd

from darli.arrays import ArrayLike
from .base import BackendBase, ConeBase, Frame, BodyInfo
from ..arrays import CasadiLikeFactory, ArrayLike
import casadi as cs


class CasadiCone(ConeBase):
    def __init__(self, force, mu, contact_type="point", X=None, Y=None):
        self.force = force
        self.mu = mu
        self.contact_type = contact_type

        # dimensions of contact surface
        self.X = X
        self.Y = Y

    def full(self) -> cs.Function:
        """
        full returns the nonlinear constraint for the contact constraint g(force)

        It should be used in the form g(force) >= 0
        """
        if self.contact_type == "point":
            return self._friction_cone(self.force)
        elif self.contact_type == "wrench":
            return self._wrench_cone(self.force)
        else:
            raise ValueError(f"Unknown contact type: {self.contact_type}")

    def linear(self):
        """
        linearized returns the linearized constraint in the form of a matrix A

        It should be used in the form A @ force <= 0
        """
        if self.contact_type == "point":
            return self._friction_cone_lin()
        elif self.contact_type == "wrench":
            return self._wrench_cone_lin()
        else:
            raise ValueError(f"Unknown contact type: {self.contact_type}")

    def _friction_cone(self, force):
        """
        nonlinear friction coloumb cone
        """

        return cs.Function(
            "nonlin_friction_cone",
            [force],
            [cs.blockcat([[force[2]], [self.mu * force[2] - cs.norm_2(force[:2])]])],
            ["force"],
            ["constraint"],
        )

    def _wrench_cone(self, force):
        """
        wrench cone constraint
        """
        assert (
            self.X is not None or self.Y is not None
        ), "X and Y of the surface must be defined"

        mu = self.mu
        X, Y = self.X, self.Y

        fx, fy, fz, tx, ty, tz = [force[i] for i in range(6)]

        tmin = (
            -mu * (X + Y) * fz + cs.fabs(Y * fx - mu * tx) + cs.fabs(X * fy - mu * ty)
        )

        tmax = mu * (X + Y) * fz - cs.fabs(Y * fx + mu * tx) - cs.fabs(X * fy + mu * ty)

        # g(force) >= 0 to satisfy the constraint
        constraints = cs.blockcat(
            [
                [(fz * mu) ** 2 - fx**2 - fy**2],
                [fz],
                [-(tx**2) + (Y * fz) ** 2],
                [-(ty**2) + (X * fz) ** 2],
                [-tmin + tz],
                [-tz + tmax],
            ]
        )

        return cs.Function(
            "nonlin_wrench_cone",
            [force],
            [constraints],
            ["force"],
            ["constraint"],
        )

    def _friction_cone_lin(self):
        """
        linearized friction coloumb cone
        """
        return cs.blockcat(
            [
                [0, 0, -1.0],
                [-1.0, 0.0, -self.mu],
                [1.0, 0.0, -self.mu],
                [0.0, -1.0, -self.mu],
                [0.0, 1.0, -self.mu],
            ]
        )

    def _wrench_cone_lin(self):
        """
        wrench cone linearized
        """
        assert (
            self.X is not None or self.Y is not None
        ), "X and Y of the surface must be defined"

        mu = self.mu
        X, Y = self.X, self.Y

        return cs.blockcat(
            [
                [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],
                [1.0, 0.0, -mu, 0.0, 0.0, 0.0],
                [0.0, -1.0, -mu, 0.0, 0.0, 0.0],
                [0.0, 1.0, -mu, 0.0, 0.0, 0.0],
                [0.0, 0.0, -Y, -1.0, 0.0, 0.0],
                [0.0, 0.0, -Y, 1.0, 0.0, 0.0],
                [0.0, 0.0, -X, 0.0, -1.0, 0.0],
                [0.0, 0.0, -X, 0.0, 1.0, 0.0],
                [-Y, -X, -(X + Y) * mu, mu, mu, -1.0],
                [-Y, X, -(X + Y) * mu, mu, -mu, -1.0],
                [Y, -X, -(X + Y) * mu, -mu, mu, -1.0],
                [Y, X, -(X + Y) * mu, -mu, -mu, -1.0],
                [Y, X, -(X + Y) * mu, mu, mu, 1.0],
                [Y, -X, -(X + Y) * mu, mu, -mu, 1.0],
                [-Y, X, -(X + Y) * mu, -mu, mu, 1.0],
                [-Y, -X, -(X + Y) * mu, -mu, -mu, 1.0],
            ]
        )


class CasadiBackend(BackendBase):
    math = CasadiLikeFactory

    def __init__(self, urdf_path: str) -> None:
        super().__init__(urdf_path)
        self.__urdf_path: str = urdf_path
        urdf = open(self.__urdf_path, "r").read()
        self.__kindyn: ckd.CasadiKinDyn = ckd.CasadiKinDyn(urdf)

        self.__nq = self.__kindyn.nq()
        self.__nv = self.__kindyn.nv()
        self.__nu = self.__nv

        self._q = cs.SX.sym("q", self.__nq)
        self._v = cs.SX.sym("v", self.__nv)
        self._dv = cs.SX.sym("dv", self.__nv)

        self._tau = cs.SX.sym("tau", self.__nv)

        self.__frame_mapping = {
            "local": ckd.CasadiKinDyn.LOCAL,
            "world": ckd.CasadiKinDyn.WORLD,
            "world_aligned": ckd.CasadiKinDyn.LOCAL_WORLD_ALIGNED,
        }

        self.__frame_types = self.__frame_mapping.keys()

    @property
    def nq(self) -> int:
        return self.__nq

    @property
    def nv(self) -> int:
        return self.__nv

    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ):
        self._q = q
        self._v = v

        if dv is not None:
            self._dv = dv
        if tau is not None:
            self._tau = tau

    def rnea(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.rnea()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )["tau"]

    def aba(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.aba()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            tau=tau if tau is not None else self._tau,
        )["a"]

    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.crba()(q=q if q is not None else self._q)["B"]

    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self.__kindyn.kineticEnergy()(
            q=q if q is not None else self._q, v=v if v is not None else self._v
        )["DT"]

    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.potentialEnergy()(q=q if q is not None else self._q)["DU"]

    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.jacobianCenterOfMass(False)(
            q=q if q is not None else self._q
        )["Jcom"]

    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return cs.jacobian(
            self.com_acc(
                q=q if q is not None else self._q,
                v=v if v is not None else self._v,
                dv=self.math.zeros(self.nv).array,
            ),
            v if v is not None else self._v,
        )

    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.centerOfMass()(
            q=q if q is not None else self._q,
            v=self.math.zeros(self.nv).array,
            a=self.math.zeros(self.nv).array,
        )["com"]

    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self.__kindyn.centerOfMass()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=self.math.zeros(self.nv).array,
        )["vcom"]

    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.centerOfMass()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )["acom"]

    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.jointTorqueRegressor()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )["regressor"]

    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.kineticEnergyRegressor()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
        )["kinetic_regressor"]

    def potential_regressor(
        self,
        q: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.potentialEnergyRegressor()(
            q=q if q is not None else self._q,
        )["potential_regressor"]

    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        if body_urdf_name is None:
            body_urdf_name = body

        return BodyInfo(
            position=self.__kindyn.fk(body_urdf_name)(q=self._q)["ee_pos"],
            rotation=self.__kindyn.fk(body_urdf_name)(q=self._q)["ee_rot"],
            jacobian={
                Frame.from_str(frame): self.__kindyn.jacobian(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q)["J"]
                for frame in self.__frame_types
            },
            lin_vel={
                Frame.from_str(frame): self.__kindyn.frameVelocity(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q, qdot=self._v)["ee_vel_linear"]
                for frame in self.__frame_types
            },
            ang_vel={
                Frame.from_str(frame): self.__kindyn.frameVelocity(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q, qdot=self._v)["ee_vel_angular"]
                for frame in self.__frame_types
            },
            lin_acc={
                Frame.from_str(frame): self.__kindyn.frameAcceleration(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q, qdot=self._v, qddot=self._dv)["ee_acc_linear"]
                for frame in self.__frame_types
            },
            ang_acc={
                Frame.from_str(frame): self.__kindyn.frameAcceleration(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q, qdot=self._v, qddot=self._dv)["ee_acc_angular"]
                for frame in self.__frame_types
            },
            djacobian={
                Frame.from_str(frame): self.__kindyn.jacobianTimeVariation(
                    body_urdf_name, self.__frame_mapping[frame]
                )(q=self._q, v=self._v)["dJ"]
                for frame in self.__frame_types
            },
        )

    def cone(
        self, force: ArrayLike | None, mu: float, type: str, X=None, Y=None
    ) -> ConeBase:
        return CasadiCone(force, mu, type, X, Y)
