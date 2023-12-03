import pinocchio as pin


class PinocchioBackend:
    def __init__(self, urdf_path) -> None:
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.nv

        # internal variables we will update to recalculate the dynamics
        self._q = pin.neutral(self.model)
        self._v = pin.utils.zero(self.nv)
        self._dv = pin.utils.zero(self.nv)
        self._tau = pin.utils.zero(self.nv)

        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

        joint_names = list(self.model.names)
        joint_names.remove("universe")
        self.joint_names = joint_names

        self._frame_mapping = {
            "local": pin.ReferenceFrame.LOCAL,
            "world": pin.ReferenceFrame.WORLD,
            "world_aligned": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        }

        self.frame_types = self._frame_mapping.keys()
        self._frames_struct = dict(zip(self._frame_mapping.keys(), 3 * [None]))

        self.mass = pin.computeTotalMass(self.model, self.data)

        self.kinetic_energy = None
        self.potential_energy = None
        self.body_position = None
        self.body_rotation = None
        self.body_jacobian = None
        self.body_jacobian_derivative = None
        self.body_linear_velocity = None
        self.body_angular_velocity = None
        self.com = None
        self.inertia_matrix = None
        # self.aba = None
        # self.rnea = None

        pin.computeAllTerms(self.model, self.data, self._q, self._v)

    def joint_iq(self, joint_name):
        return self.model.getJointId(joint_name)

    def rnea_fn(self, q, v, dv):
        return pin.rnea(self.model, self.data, q, v, dv)

    def aba_fn(self, q, v, tau):
        return pin.aba(self.model, self.data, q, v, tau)

    def update(self, q, v, dv=None, tau=None):
        self._q = q
        self._v = v

        if dv is not None:
            self._dv = dv
            self._tau = self.rnea_fn(self._q, self._v, self._dv)
        if tau is not None:
            self._tau = tau
            self._dv = self.aba_fn(self._q, self._v, tau)

        pin.computeAllTerms(self.model, self.data, self._q, self._v)

        self.com = {
            "position": self.data.com[0],
            "velocity": self.data.vcom[0],
            "acceleration": self.data.acom[0],
            "jacobian": self.data.Jcom,
            "jacobian_dt": None,  # TODO: actually, there is no way to compute it directly from pinocchio
        }

        self.kinetic_energy = self.data.kinetic_energy
        self.potential_energy = self.data.potential_energy

        self.inertia_matrix = self.data.M
        # self.rnea = self.data.tau  # inverse dynamics solution
        # self.aba = self.data.ddq  # forward dynamics solution

    def update_body(self, body, body_urdf_name=None):
        if body_urdf_name is None:
            body_urdf_name = body

        frame_idx = self.model.getFrameId(body_urdf_name)
        frame = self.data.oMf[frame_idx]

        self.body_position = frame.translation
        self.body_rotation = frame.rotation

        self.body_jacobian = {}
        self.body_jacobian_derivative = {}
        self.body_linear_velocity = {}
        self.body_angular_velocity = {}
        self.body_linear_acceleration = {}
        self.body_angular_acceleration = {}

        for ref_frame, pin_frame in self._frame_mapping.items():
            self.body_jacobian[ref_frame] = pin.getFrameJacobian(
                self.model, self.data, frame_idx, pin_frame
            )

            self.body_linear_velocity[ref_frame] = (
                self.body_jacobian[ref_frame][:3, :] @ self._v
            )
            self.body_angular_velocity[ref_frame] = (
                self.body_jacobian[ref_frame][3:, :] @ self._v
            )

            self.body_jacobian_derivative[
                ref_frame
            ] = pin.getFrameJacobianTimeVariation(
                self.model, self.data, frame_idx, pin_frame
            )

            self.body_linear_acceleration[ref_frame] = (
                self.body_jacobian_derivative[ref_frame][:3, :] @ self._v
                + self.body_jacobian[ref_frame][:3, :] @ self._dv
            )

            self.body_angular_acceleration[ref_frame] = (
                self.body_jacobian_derivative[ref_frame][3:, :] @ self._v
                + self.body_jacobian[ref_frame][3:, :] @ self._dv
            )
