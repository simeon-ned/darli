import pinocchio as pin

from ._base import BackendBase, BodyInfo, ConeBase, Frame, JointType, CentroidalDynamics
from ..utils.arrays import ArrayLike, NumpyLikeFactory
import numpy as np
import numpy.typing as npt
from typing import Dict


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

    def __init__(
        self,
        urdf_path: str,
        root_joint: JointType | None = None,
        fixed_joints: Dict[str, float | npt.ArrayLike] = None,
    ) -> None:
        super().__init__(urdf_path, root_joint, fixed_joints)

        # backend base already contains model and data in pinocchio format
        # as we have created everything in super().__init__(), we can use it here
        self.__model: pin.Model = self._pinmodel
        self.__data: pin.Data = self._pindata

        self.__nq = self.__model.nq
        self.__nv = self.__model.nv
        self.__nu = self.__nv

        self._q = pin.neutral(self.__model)
        self._v = pin.utils.zero(self.__nv)
        self._dv = pin.utils.zero(self.__nv)

        self._tau = pin.utils.zero(self.__nv)

        self.__frame_mapping = {
            "local": pin.LOCAL,
            "world": pin.WORLD,
            "world_aligned": pin.LOCAL_WORLD_ALIGNED,
        }

        self.__frame_types = self.__frame_mapping.keys()

        self.__body_info_cache = {}
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
        self._q = q
        self._v = v

        if dv is not None:
            self._dv = dv
            # update _tau only if it not given
            if tau is None:
                self._tau = pin.rnea(
                    self.__model, self.__data, self._q, self._v, self._dv
                )
        if tau is not None:
            self._tau = tau
            # update _dv only if it not given
            if dv is None:
                self._dv = pin.aba(
                    self.__model, self.__data, self._q, self._v, self._tau
                )

        pin.computeAllTerms(self.__model, self.__data, self._q, self._v)
        pin.jacobianCenterOfMass(self.__model, self.__data, self._q)
        pin.computeForwardKinematicsDerivatives(
            self.__model, self.__data, self._q, self._v, self._dv
        )
        self.__dJcom_dt = pin.getCenterOfMassVelocityDerivatives(
            self.__model, self.__data
        )

        if dv is not None or tau is not None:
            # we have to calculate centerOfMass only if we computed dv previously
            pin.centerOfMass(self.__model, self.__data, self._q, self._v, self._dv)
            # compute centroidal dynamics and first derivative
            pin.computeCentroidalMomentumTimeVariation(
                self.__model, self.__data, self._q, self._v, self._dv
            )
            # compute jacobians of centroidal dynamics
            self.__centroidal_derivatives = pin.computeCentroidalDynamicsDerivatives(
                self.__model, self.__data, self._q, self._v, self._dv
            )

        # we have to clear body info cache
        self.__body_info_cache = {}

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
        return pin.aba(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
            tau if tau is not None else self._tau,
        )

    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        if q is None:
            return self.__data.M

        self._q = q
        pin.computeAllTerms(self.__model, self.__data, q, self._v)
        return self.__data.M

    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        if q is None and v is None:
            return self.__data.kinetic_energy

        self._q = q
        self._v = v
        pin.computeAllTerms(self.__model, self.__data, q, v)
        return self.__data.kinetic_energy

    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        if q is None:
            return self.__data.potential_energy

        self._q = q
        pin.computeAllTerms(self.__model, self.__data, q, self._v)
        return self.__data.potential_energy

    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        if q is None:
            return self.__data.Jcom

        self._q = q
        pin.jacobianCenterOfMass(self.__model, self.__data, self._q)
        return self.__data.Jcom

    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        if q is None and v is None:
            return self.__dJcom_dt

        self._q = q
        self._v = v
        pin.centerOfMass(self.__model, self.__data, self._q, self._v, self._dv)
        self.__dJcom_dt = pin.getCenterOfMassVelocityDerivatives(
            self.__model, self.__data
        )
        return self.__dJcom_dt

    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        if q is None:
            return self.__data.com[0]

        self._q = q

        pin.centerOfMass(self.__model, self.__data, self._q, self._v, self._dv)
        return self.__data.com[0]

    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        if q is None and v is None:
            return self.__data.vcom[0]

        self._q = q if q is not None else self._q
        self._v = v if v is not None else self._v
        pin.centerOfMass(self.__model, self.__data, self._q, self._v, self._dv)
        return self.__data.vcom[0]

    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        if q is None and v is None and dv is None:
            return self.__data.acom[0]

        self._q = q if q is not None else self._q
        self._v = v if v is not None else self._v
        self._dv = dv if dv is not None else self._dv
        pin.centerOfMass(self.__model, self.__data, self._q, self._v, self._dv)
        return self.__data.acom[0]

    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        if q is None and v is None and dv is None:
            return self.__data.jointTorqueRegressor

        self._q = q
        self._v = v
        self._dv = dv

        pin.computeAllTerms(self.__model, self.__data, q, v)
        pin.computeJointTorqueRegressor(self.__model, self.__data, q, v, dv)

        return self.__data.jointTorqueRegressor

    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ) -> ArrayLike:
        if q is not None or v is not None:
            self._q = q if q is not None else self._q
            self._v = v if v is not None else self._v
            pin.computeAllTerms(self.__model, self.__data, q, v)

        regressor = np.zeros((1, self.nbodies * 10))
        for i in range(self.nbodies):
            vel = pin.getVelocity(self.__model, self.__data, i + 1, pin.LOCAL)
            vl = vel.linear
            va = vel.angular

            regressor[0, i * 10 + 0] = 0.5 * (vl[0] ** 2 + vl[1] ** 2 + vl[2] ** 2)
            regressor[0, i * 10 + 1] = -va[1] * vl[2] + va[2] * vl[1]
            regressor[0, i * 10 + 2] = va[0] * vl[2] - va[2] * vl[0]
            regressor[0, i * 10 + 3] = -va[0] * vl[1] + va[1] * vl[0]
            regressor[0, i * 10 + 4] = 0.5 * va[0] ** 2
            regressor[0, i * 10 + 5] = va[0] * va[1]
            regressor[0, i * 10 + 6] = 0.5 * va[1] ** 2
            regressor[0, i * 10 + 7] = va[0] * va[2]
            regressor[0, i * 10 + 8] = va[1] * va[2]
            regressor[0, i * 10 + 9] = 0.5 * va[2] ** 2

        return regressor

    def potential_regressor(
        self,
        q: ArrayLike | None = None,
    ) -> ArrayLike:
        if q is not None:
            self._q = q if q is not None else self._q
            pin.computeAllTerms(self.__model, self.__data, q, self._v)

        regressor = np.zeros((1, self.nbodies * 10))
        for i in range(self.nbodies):
            r = self.__data.oMi[i + 1].translation
            R = self.__data.oMi[i + 1].rotation
            g = -self.__model.gravity.linear

            res = R.T @ g
            regressor[0, i * 10 + 0] = g.dot(r)
            regressor[0, i * 10 + 1] = res[0]
            regressor[0, i * 10 + 2] = res[1]
            regressor[0, i * 10 + 3] = res[2]

        return regressor

    def _spatial_kinetic_energy_jacobian(self, v, w):
        # spatial kinetic energy regressor jacobian with respect to velocity and angular velocityf
        jacobian = np.array(
            [
                [v[0], v[1], v[2], 0, 0, 0],
                [0, w[2], -w[1], 0, -v[2], v[1]],
                [-w[2], 0, w[0], v[2], 0, -v[0]],
                [w[1], -w[0], 0, -v[1], v[0], 0],
                [0, 0, 0, w[0], 0, 0],
                [0, 0, 0, w[1], w[0], 0],
                [0, 0, 0, 0, w[1], 0],
                [0, 0, 0, w[2], 0, w[0]],
                [0, 0, 0, 0, w[2], w[1]],
                [0, 0, 0, 0, 0, w[2]],
            ]
        )
        return jacobian.T

    def momentum_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """momentum_regressor computes parametric momentum and
            partial derivative of lagrangian with respect to configuration

        Args:
            q (ArrayLike | None, optional): _description_. Defaults to None.
            v (ArrayLike | None, optional): _description_. Defaults to None.

        Returns:
            tuple[ArrayLike, ArrayLike]: momentum regressor and partial derivative of lagrangian regressor
        """
        if q is not None or v is not None:
            self._q = q if q is not None else self._q
            self._v = v if v is not None else self._v
            pin.computeAllTerms(self.__model, self.__data, q, v)

        # static regressor Y(q, 0, 0)
        pin.computeJointTorqueRegressor(
            self.__model,
            self.__data,
            self._q,
            np.zeros(self.nv),
            np.zeros(self.nv),
        )
        static = self.__data.jointTorqueRegressor.copy()

        # phi_p = M(q) v = (Y(q, 0, v) - Y(q, 0, 0)) @ theta
        pin.computeJointTorqueRegressor(
            self.__model,
            self.__data,
            self._q,
            np.zeros(self.nv),
            self._v,
        )
        phi_p = self.__data.jointTorqueRegressor.copy() - static.copy()

        # compute the derivative
        dphi_h = np.zeros_like(phi_p)
        for i in range(self.nbodies):
            spatial_vel = pin.getVelocity(
                self._pinmodel,
                self._pindata,
                i + 1,
                pin.LOCAL,
            )
            velocity_derivatives = pin.getJointVelocityDerivatives(
                self._pinmodel,
                self._pindata,
                i + 1,
                pin.LOCAL,
            )

            dvb_dv = velocity_derivatives[1].copy()
            phik_dv = self._spatial_kinetic_energy_jacobian(
                spatial_vel.linear.copy(), spatial_vel.angular.copy()
            )
            phik_dv_joint = dvb_dv.T @ phik_dv
            dphi_h[:, i * 10 : (i + 1) * 10] = phik_dv_joint.copy()

        dphi_h -= static

        return phi_p, dphi_h

    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        if body_urdf_name is None:
            body_urdf_name = body

        # if we have cached information about body, clean it
        if body_urdf_name in self.__body_info_cache:
            return self.__body_info_cache[body_urdf_name]

        frame_idx = self.__model.getFrameId(body_urdf_name)

        jacobian = {}
        djacobian = {}
        lin_vel = {}
        ang_vel = {}
        lin_acc = {}
        ang_acc = {}
        for frame_str, fstr in self.__frame_mapping.items():
            frame = Frame.from_str(frame_str)

            jacobian[frame] = pin.getFrameJacobian(
                self.__model, self.__data, frame_idx, fstr
            )
            djacobian[frame] = pin.getFrameJacobianTimeVariation(
                self.__model, self.__data, frame_idx, fstr
            )
            lin_vel[frame] = jacobian[frame][:3] @ self._v
            ang_vel[frame] = jacobian[frame][3:] @ self._v
            lin_acc[frame] = (
                jacobian[frame][:3] @ self._dv + djacobian[frame][:3] @ self._v
            )
            ang_acc[frame] = (
                jacobian[frame][3:] @ self._dv + djacobian[frame][3:] @ self._v
            )

        result = BodyInfo(
            position=self.__data.oMf[frame_idx].translation,
            rotation=self.__data.oMf[frame_idx].rotation,
            quaternion=pin.se3ToXYZQUAT(self.__data.oMf[frame_idx])[3:],
            jacobian=jacobian,
            djacobian=djacobian,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            lin_acc=lin_acc,
            ang_acc=ang_acc,
        )
        self.__body_info_cache[body_urdf_name] = result

        return result

    def cone(
        self, force: ArrayLike | None, mu: float, type: str, X=None, Y=None
    ) -> ConeBase:
        return PinocchioCone(force, mu, type, X, Y)

    def integrate_configuration(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dt: float = 1.0,
    ) -> ArrayLike:
        return pin.integrate(
            self.__model,
            q if q is not None else self._q,
            v * dt if v is not None else self._v * dt,
        )

    def centroidal_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CentroidalDynamics:
        if q is None and v is None and dv is None:
            return CentroidalDynamics(
                matrix=self.__data.Ag,
                linear=self.__data.hg.linear,
                angular=self.__data.hg.angular,
                linear_dt=self.__data.dhg.linear,
                angular_dt=self.__data.dhg.angular,
                matrix_dt=self.__centroidal_derivatives[0],
                dynamics_jacobian_q=self.__centroidal_derivatives[1],
                dynamics_jacobian_v=self.__centroidal_derivatives[2],
                dynamics_jacobian_dv=self.__centroidal_derivatives[3],
            )

        self._q = q if q is not None else self._q
        self._v = v if v is not None else self._v
        self._dv = dv if dv is not None else self._dv

        pin.computeCentroidalMomentumTimeVariation(
            self.__model, self.__data, self._q, self._v, self._dv
        )
        pin.computeCentroidalDynamicsDerivatives(
            self.__model, self.__data, self._q, self._v, self._dv
        )

        return CentroidalDynamics(
            matrix=self.__data.Ag,
            linear=self.__data.hg.linear,
            angular=self.__data.hg.angular,
            linear_dt=self.__data.dhg.linear,
            angular_dt=self.__data.dhg.angular,
            matrix_dt=self.__centroidal_derivatives[0],
            dynamics_jacobian_q=self.__centroidal_derivatives[1],
            dynamics_jacobian_v=self.__centroidal_derivatives[2],
            dynamics_jacobian_dv=self.__centroidal_derivatives[3],
        )
