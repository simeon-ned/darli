import pinocchio.casadi as cpin
import pinocchio as pin
from .liecasadi import SO3
from ._base import BackendBase, ConeBase
from ._structs import Frame, BodyInfo, CentroidalDynamics, JointType
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
        description_path: str,
        root_joint: JointType | None = JointType.OMIT,
        fixed_joints: Dict[str, float | npt.ArrayLike] = None,
    ) -> None:
        super().__init__(description_path, root_joint, fixed_joints)

        self.__model: cpin.Model = cpin.Model(self._pinmodel)
        self.__data: cpin.Data = self.__model.createData()

        self.__nq = self.__model.nq
        self.__nv = self.__model.nv
        self.__nu = self.__nv

        self._q = cs.SX.sym("q", self.__nq)
        self._v = cs.SX.sym("v", self.__nv)
        self._dv = cs.SX.sym("dv", self.__nv)

        self._tau = cs.SX.sym("tau", self.__nv)

        self.__frame_mapping = {
            "local": pin.LOCAL,
            "world": pin.WORLD,
            "world_aligned": pin.LOCAL_WORLD_ALIGNED,
        }

        self.__frame_types = self.__frame_mapping.keys()
        self.__centroidal_derivatives = None

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

        if dv is not None or tau is not None:
            self.__centroidal_derivatives = cpin.computeCentroidalDynamicsDerivatives(
                self.__model, self.__data, self._q, self._v, self._dv
            )

    def rnea(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return cpin.rnea(
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
        return cpin.aba(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
            tau if tau is not None else self._tau,
        )

    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        return cpin.crba(self.__model, self.__data, q if q is not None else self._q)

    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        return cpin.computeKineticEnergy(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
        )

    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        return cpin.computePotentialEnergy(
            self.__model, self.__data, q if q is not None else self._q
        )

    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        return cpin.jacobianCenterOfMass(
            self.__model, self.__data, q if q is not None else self._q
        )

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
        return cpin.centerOfMass(
            self.__model, self.__data, q if q is not None else self._q
        )

    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        cpin.centerOfMass(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
        )

        return self.__data.vcom[0]

    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        cpin.centerOfMass(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
            dv if dv is not None else self._dv,
        )

        return self.__data.acom[0]

    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        return cpin.computeJointTorqueRegressor(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
            dv if dv is not None else self._dv,
        )

    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ) -> ArrayLike:
        return cpin.computeKineticEnergyRegressor(
            self.__model,
            self.__data,
            q if q is not None else self._q,
            v if v is not None else self._v,
        )

    def potential_regressor(
        self,
        q: ArrayLike | None = None,
    ) -> ArrayLike:
        return cpin.computePotentialEnergyRegressor(
            self.__model,
            self.__data,
            q if q is not None else self._q,
        )

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
        raise NotImplementedError("This function is not implemented yet")
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

    def update_body(self, body: str, body_urdf_name: str | None = None) -> BodyInfo:
        if body_urdf_name is None:
            body_urdf_name = body

        # check that the frame is present in the model
        if not self.__model.existFrame(body_urdf_name):
            raise KeyError(f"Frame {body_urdf_name} does not exist in the model")

        frame_idx = self.__model.getFrameId(body_urdf_name)

        jacobian = {}
        djacobian = {}
        lin_vel = {}
        ang_vel = {}
        lin_acc = {}
        ang_acc = {}
        cpin.framesForwardKinematics(self.__model, self.__data, self._q)
        for frame_str, fstr in self.__frame_mapping.items():
            frame = Frame.from_str(frame_str)

            jacobian[frame] = cpin.getFrameJacobian(
                self.__model, self.__data, frame_idx, fstr
            )
            djacobian[frame] = cpin.getFrameJacobianTimeVariation(
                self.__model, self.__data, frame_idx, fstr
            )
            # TODO: Redo this with standart pin functions
            lin_vel[frame] = jacobian[frame][:3, :] @ self._v
            ang_vel[frame] = jacobian[frame][3:, :] @ self._v
            lin_acc[frame] = (
                jacobian[frame][:3, :] @ self._dv + djacobian[frame][:3, :] @ self._v
            )
            ang_acc[frame] = (
                jacobian[frame][3:, :] @ self._dv + djacobian[frame][3:, :] @ self._v
            )

        result = BodyInfo(
            position=self.__data.oMf[frame_idx].translation,
            rotation=self.__data.oMf[frame_idx].rotation,
            quaternion=cpin.SE3ToXYZQUAT(self.__data.oMf[frame_idx])[3:],
            jacobian=jacobian,
            djacobian=djacobian,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            lin_acc=lin_acc,
            ang_acc=ang_acc,
        )
        # DO WE NEED THIS IN CASADI BACKEND?
        # self.__body_info_cache[body_urdf_name] = result

        return result

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
        return cpin.integrate(self.__model, q, v * dt)

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

        cpin.computeCentroidalMomentumTimeVariation(
            self.__model, self.__data, self._q, self._v, self._dv
        )
        cpin.computeCentroidalDynamicsDerivatives(
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
