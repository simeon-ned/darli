import pinocchio as pin
from .base import BackendBase, Frame, BodyInfo, ConeBase
from ..arrays import ArrayLike, NumpyLikeFactory
import numpy as np


class PinocchioCone(ConeBase):
    def __init__(self, force, mu, contact_type="point", X=None, Y=None):
        self.force = force
        self.mu = mu
        self.contact_type = contact_type

        # dimensions of contact surface
        self.X = X
        self.Y = Y

    def full(self) -> np.ndarray:
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

        return np.block([[force[2]], [self.mu * force[2] - np.linalg.norm(force[:2])]])

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

        tmin = -mu * (X + Y) * fz + np.abs(Y * fx - mu * tx) + np.abs(X * fy - mu * ty)

        tmax = mu * (X + Y) * fz - np.abs(Y * fx + mu * tx) - np.abs(X * fy + mu * ty)

        # g(force) >= 0 to satisfy the constraint
        constraints = np.block(
            [
                [(fz * mu) ** 2 - fx**2 - fy**2],
                [fz],
                [-(tx**2) + (Y * fz) ** 2],
                [-(ty**2) + (X * fz) ** 2],
                [-tmin + tz],
                [-tz + tmax],
            ]
        )

        return constraints

    def _friction_cone_lin(self):
        """
        linearized friction coloumb cone
        """
        return np.block(
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

        return np.block(
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

    def cone(
        self, force: ArrayLike | None, mu: float, type: str, X=None, Y=None
    ) -> ConeBase:
        return PinocchioCone(force, mu, type, X, Y)
