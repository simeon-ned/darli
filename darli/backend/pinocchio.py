import pinocchio as pin
from .base import BackendBase, Frame, BodyInfo
from ..arrays import ArrayLike, NumpyLikeFactory


class PinocchioBackend(BackendBase):
    math = NumpyLikeFactory

    def __init__(self, urdf_path: str) -> None:
        self.__urdf_path: str = urdf_path
        self.__model: pin.Model = pin.buildModelFromUrdf(self.__urdf_path)
        self.__data: pin.Data = self.__model.createData()

        self.__nq = self.__model.nq
        self.__nv = self.__model.nv
        self.__nu = self.__nv

        self._q = pin.neutral(self.__model)
        self._v = pin.utils.zero(self.__nv)
        self._dv = pin.utils.zero(self.__nv)

        self._tau = pin.utils.zero(self.__nv)

        self.update(self._q, self._v, self._dv, self._tau)

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
        assert dv is not None or tau is not None, "Either dv or tau must be provided"
        self._q = q
        self._v = v

        if dv is not None:
            self._dv = dv
            self._tau = pin.rnea(self.__model, self.__data, self._q, self._v, self._dv)
        if tau is not None:
            self._tau = tau
            self._dv = pin.aba(self.__model, self.__data, self._q, self._v, self._tau)

        pin.computeAllTerms(self.__model, self.__data, self._q, self._v)

    def rnea(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return pin.rnea(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
            dv if dv is not None else self._dv,
        )

    def aba(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        if q is None:
            return self.__data.M

        self._q = q
        pin.computeAllTerms(self.__model, self.__data, q, self._v)
        return self.__data.M

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

    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        raise NotImplementedError

    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        raise NotImplementedError

    def potential_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        raise NotImplementedError

    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        pass
