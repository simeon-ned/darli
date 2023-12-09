import casadi_kin_dyn.casadi_kin_dyn as ckd
from .base import BackendBase, Frame, BodyInfo
from ..arrays import CasadiLikeFactory, ArrayLike
import casadi as cs


class CasadiBackend(BackendBase):
    math = CasadiLikeFactory

    def __init__(self, urdf_path: str) -> None:
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
        pass

    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.crba()(q=q if q is not None else self._q)["B"]

    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self.__kindyn.kinetic_energy()(
            q=q if q is not None else self._q, v=v if v is not None else self._v
        )

    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.potential_energy()(q=q if q is not None else self._q)

    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.jacobianCenterOfMass(False)(
            q=q if q is not None else self._q
        )

    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        raise NotImplementedError

    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        return self.__kindyn.centerOfMass()(
            q=q if q is not None else self._q,
            v=self.math.zeros(self.nv),
            a=self.math.zeros(self.nv),
        )

    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return self.__kindyn.centerOfMass()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=self.math.zeros(self.nv),
        )

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
        )

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
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.kineticEnergyRegressor()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )["regressor"]

    def potential_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return self.__kindyn.potentialEnergyRegressor()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )["regressor"]

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
