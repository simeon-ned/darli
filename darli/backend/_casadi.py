import casadi_kin_dyn.casadi_kin_dyn as ckd

from .liecasadi import SO3
from ._base import BackendBase, ConeBase, Frame, BodyInfo, JointType, CentroidalDynamics
from ..utils.arrays import CasadiLikeFactory, ArrayLike
import casadi as cs
from typing import Dict
import numpy.typing as npt
from . import liecasadi as lie


# TODO: Parse joints tyoe from description, and create
# approprate Lie group for them


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

    def __init__(
        self,
        urdf_path: str,
        root_joint: JointType | None = JointType.OMIT,
        fixed_joints: Dict[str, float | npt.ArrayLike] = None,
    ) -> None:
        super().__init__(urdf_path)
        if not fixed_joints:
            fixed_joints = {}

        self.__joint_types = {
            JointType.FREE_FLYER: ckd.CasadiKinDyn.JointType.FREE_FLYER,
            JointType.PLANAR: ckd.CasadiKinDyn.JointType.PLANAR,
            JointType.OMIT: ckd.CasadiKinDyn.JointType.OMIT,
        }

        self.__urdf_path: str = urdf_path
        urdf = open(self.__urdf_path, "r").read()
        self.__kindyn: ckd.CasadiKinDyn = ckd.CasadiKinDyn(
            urdf,
            root_joint=self.__joint_types[root_joint],
            fixed_joints=fixed_joints,
        )

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

    def _spatial_kinetic_energy_jacobian(self):
        # Define CasADi symbolic variables
        v = cs.SX.sym("v", 3)
        w = cs.SX.sym("w", 3)

        # Define the Jacobian matrix as a CasADi SX matrix
        jacobian = cs.SX.zeros(10, 6)

        jacobian[0, :] = cs.vertcat(v[0], v[1], v[2], 0, 0, 0)
        jacobian[1, :] = cs.vertcat(0, w[2], -w[1], 0, -v[2], v[1])
        jacobian[2, :] = cs.vertcat(-w[2], 0, w[0], v[2], 0, -v[0])
        jacobian[3, :] = cs.vertcat(w[1], -w[0], 0, -v[1], v[0], 0)
        jacobian[4, :] = cs.vertcat(0, 0, 0, w[0], 0, 0)
        jacobian[5, :] = cs.vertcat(0, 0, 0, w[1], w[0], 0)
        jacobian[6, :] = cs.vertcat(0, 0, 0, 0, w[1], 0)
        jacobian[7, :] = cs.vertcat(0, 0, 0, w[2], 0, w[0])
        jacobian[8, :] = cs.vertcat(0, 0, 0, 0, w[2], w[1])
        jacobian[9, :] = cs.vertcat(0, 0, 0, 0, 0, w[2])

        # Transpose the Jacobian matrix
        jacobian_transposed = jacobian.T

        # Define the CasADi function
        spatial_kinetic_energy_jacobian = cs.Function(
            "spatial_kinetic_energy_jacobian",
            [v, w],
            [jacobian_transposed],
        )

        return spatial_kinetic_energy_jacobian

    def momentum_regressor(
        self,
        q_inp: ArrayLike | None = None,
        v_inp: ArrayLike | None = None,
    ):
        # store functions
        spatial_kinetic_energy_jacobian = self._spatial_kinetic_energy_jacobian()
        torque_reg_fn = self.__kindyn.jointTorqueRegressor()

        # static regressor Y(q, 0, 0)
        static = torque_reg_fn(
            q=self._q,
            v=cs.SX.zeros(self.nv),
            a=cs.SX.zeros(self.nv),
        )["regressor"]

        # phi_p = M(q) @ v = (Y(q, 0, v) - Y(q, 0, 0)) @ v
        phi_p = (
            torque_reg_fn(
                q=self._q,
                v=cs.SX.zeros(self.nv),
                a=self._v,
            )["regressor"]
            - static
        )

        # compute the partial derivative of lagrangian w.r.t. configuration
        dphi_h = cs.SX.zeros(*static.shape)

        joint_idx = 0
        for idx, joint_name in enumerate(self.joint_names):
            if self.__kindyn.joint_nq(joint_name) == 0:
                continue
            # find the joint index
            body_urdf_name = self.__kindyn.parentLink(joint_name)

            # find spatial velocity
            lin_vel = self.__kindyn.frameVelocity(
                body_urdf_name, self.__frame_mapping["local"]
            )(q=self._q, qdot=self._v)["ee_vel_linear"]

            ang_vel = self.__kindyn.frameVelocity(
                body_urdf_name, self.__frame_mapping["local"]
            )(q=self._q, qdot=self._v)["ee_vel_angular"]

            # compute the spatial kinetic energy jacobian
            phik_dv = spatial_kinetic_energy_jacobian(lin_vel, ang_vel)
            # compute the velocity derivatives with respect to configuration
            dvb_dv = self.__kindyn.jointVelocityDerivatives(
                body_urdf_name, self.__frame_mapping["local"]
            )(
                q=self._q,
                v=self._v,
            )["v_partial_dv"]

            phik_dv_joint = dvb_dv.T @ phik_dv
            dphi_h[:, joint_idx * 10 : (joint_idx + 1) * 10] = phik_dv_joint
            joint_idx += 1

        dphi_h -= static

        return cs.Function(
            "momentum_regressor",
            [self._q, self._v],
            [phi_p, dphi_h],
            ["q", "v"],
            ["phi_p", "dphi_h"],
        )(
            q_inp if q_inp is not None else self._q,
            v_inp if v_inp is not None else self._v,
        )

    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        if body_urdf_name is None:
            body_urdf_name = body
        return BodyInfo(
            position=self.__kindyn.fk(body_urdf_name)(q=self._q)["ee_pos"],
            rotation=self.__kindyn.fk(body_urdf_name)(q=self._q)["ee_rot"],
            quaternion=SO3.from_matrix(
                self.__kindyn.fk(body_urdf_name)(q=self._q)["ee_rot"]
            ).xyzw,
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

    def integrate_configuration(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dt: float | cs.SX = 1.0,
    ) -> ArrayLike:
        if self.nq != self.nv:
            q = q if q is not None else self._q
            v = v if v is not None else self._v

            # we have to use lie geometry
            # to integrate se3 and joint space separately
            # TODO:
            # replace with SE3
            # what if someone will fix base position?
            pos = q[:3]
            xyzw = q[3:7]
            so3 = lie.SO3(xyzw)
            joints = q[7:]

            pos_tang = v[:3] * dt
            so3_tang = lie.SO3Tangent(v[3:6] * dt)
            joint_tang = v[6:] * dt

            # se3 = lie.SE3(pos=pos, xyzw=xyzw)
            # se3_tang = lie.SE3Tangent(v[:6] * dt)
            # container[:3] = se3_next.xyzw
            # container[3:7] = se3_next.xyzw
            # se3_next = se3 + so3_tang

            pos_next = pos + pos_tang
            so3_next = so3 + so3_tang
            joints_next = joints + joint_tang

            container = cs.SX.zeros(self.nq)
            configuration_next = cs.vertcat(pos_next, so3_next.xyzw, joints_next)
            # container[:3] = pos_next
            # container[3:7] = so3_next.xyzw
            # container[7:] = joints_next
            container = configuration_next
            return container
        else:
            return (q if q is not None else self._q) + (
                v if v is not None else self._v
            ) * dt

        # do not ever try to use this in optimization
        # return self.__kindyn.integrate()(
        #     q=q if q is not None else self._q,
        #     v=v * dt if v is not None else self._v * dt,
        # )["qnext"]

    def centroidal_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CentroidalDynamics:
        dyn = self.__kindyn.computeCentroidalDynamics()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )
        dyn_der = self.__kindyn.computeCentroidalDynamicsDerivatives()(
            q=q if q is not None else self._q,
            v=v if v is not None else self._v,
            a=dv if dv is not None else self._dv,
        )

        return CentroidalDynamics(
            matrix=dyn["Ag"],
            linear=dyn["h_lin"],
            angular=dyn["h_ang"],
            linear_dt=dyn["dh_lin"],
            angular_dt=dyn["dh_ang"],
            matrix_dt=dyn_der["dh_dq"],
            dynamics_jacobian_q=dyn_der["dhdot_dq"],
            dynamics_jacobian_v=dyn_der["dhdot_dv"],
            dynamics_jacobian_dv=dyn_der["dhdot_da"],
        )
