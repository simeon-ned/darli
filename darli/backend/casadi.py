import casadi_kin_dyn.casadi_kin_dyn as ckd
from .base import BackendBase, Frame, BodyInfo
from ..arrays import CasadiLike, CasadiLikeFactory, ArrayLike
import casadi as cs


class CasadiBackend(BackendBase):
    array_factory = CasadiLikeFactory

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
    ) -> ArrayLike:
        pass

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
        pass

    def potential_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        pass
