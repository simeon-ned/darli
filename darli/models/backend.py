import casadi_kin_dyn.casadi_kin_dyn as cas_kin_dyn
import casadi as cs

# BUG:
# THE DICTIONARY OF JACOBIANS FOR BODIES AND VELOCITIES ARE REPEATED


# STRUCTURE:
# ModelBackend
# Functions for aba, rnea, fk etc, basic arguments of model, interface to urdf
# RobotModel - interface to forward_dynamics, inverse_dynamics, etc
#

# TODO:
# //////////////////////////////////////////////////////////////////
# ADD EFFORT LIMITS FROM URDF
# DOC STRING
# ADD FRICTION AND JOINT INERTIAS
# CONVERSION TO QUATERNIONS
# ADD NAMES AND MAP OF ACTIVE JOINTS
# MOVE BODY ATTRIBUTE TO SEPARATE CLASS WITH OWN ATTRIBUTES
# MOVE CONTACTS TO SEPARATE CLASS WITH OWN ATTRIBUTES
# ADD FRICTION MODEL
# ADD MODEL BACKEND AS INPUT TO MODEL, SUCH THAT IT BECOME INDEPENDENT FROM kindyn
# ADD QUATERNION DERIVATIVE TO STATE SPACE REPRESENTATION
# PUT REGRESSORS TO BACK
#


class KinDynBackend:
    def __init__(self, urdf_path) -> None:
        # pass

        self.urdf_path = urdf_path
        urdf = open(self.urdf_path, "r").read()
        self._kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

        self.nq = self._kindyn.nq()
        self.nv = self._kindyn.nv()
        self.nu = self.nv

        # internal casadi variables to generate functions
        self._q = cs.SX.sym("q", self.nq)
        self._v = cs.SX.sym("v", self.nv)
        self._dv = cs.SX.sym("dv", self.nv)

        self._tau = cs.SX.sym("tau", self.nv)

        self.q_min = self._kindyn.q_min()
        self.q_max = self._kindyn.q_max()

        joint_names = self._kindyn.joint_names()
        joint_names.remove("universe")
        self.joint_names = joint_names
        self.joint_iq = self._kindyn.joint_iq

        # TODO: Add bodies names
        bodies_names = None
        self.nb = None

        # mappings
        self._frame_mapping = {
            "local": cas_kin_dyn.CasadiKinDyn.LOCAL,
            "world": cas_kin_dyn.CasadiKinDyn.WORLD,
            "world_aligned": cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED,
        }

        self.frame_types = self._frame_mapping.keys()
        self._frames_struct = dict(zip(self._frame_mapping.keys(), 3 * [None]))

        self.mass = self._kindyn.mass()

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
        self.aba = None
        self.rnea = None

        self.update_dynamics()

    def update_body(self, body, body_urdf_name=None):
        # print(body, urdf_name)
        if body_urdf_name is None:
            body_urdf_name = body

        fk = self._kindyn.fk(body_urdf_name)

        position = fk(q=self._q)["ee_pos"]
        self.body_position = cs.Function(
            f"position_{body}", [self._q], [position], ["q"], ["position"]
        )

        rotation = fk(q=self._q)["ee_rot"]
        self.body_rotation = cs.Function(
            f"rotation_{body}", [self._q], [rotation], ["q"], ["rotation"]
        )

        # kindyn.jacobian(frame, force_reference_frame)
        self.body_jacobian = {}
        self.body_jacobian_derivative = {}
        self.body_linear_velocity = {}
        self.body_angular_velocity = {}
        self.body_linear_acceleration = {}
        self.body_angular_acceleration = {}

        for reference_frame in self.frame_types:
            jacobian = self._kindyn.jacobian(
                body_urdf_name, self._frame_mapping[reference_frame]
            )
            self.body_jacobian[reference_frame] = cs.Function(
                f"{body}_jacobian_{reference_frame}",
                [self._q],
                [jacobian(q=self._q)["J"]],
                ["q"],
                ["jacobian"],
            )

            dfk = self._kindyn.frameVelocity(
                body_urdf_name, self._frame_mapping[reference_frame]
            )
            self.body_linear_velocity[reference_frame] = cs.Function(
                f"{body}_linear_velocity_{reference_frame}",
                [self._q, self._v],
                [dfk(q=self._q, qdot=self._v)["ee_vel_linear"]],
                ["q", "v"],
                [f"linear_velocity_{reference_frame}"],
            )

            self.body_angular_velocity[reference_frame] = cs.Function(
                f"{body}_angular_velocity_{reference_frame}",
                [self._q, self._v],
                [dfk(q=self._q, qdot=self._v)["ee_vel_angular"]],
                ["q", "v"],
                [f"angular_velocity_{reference_frame}"],
            )

            ddfk = self._kindyn.frameAcceleration(
                body_urdf_name, self._frame_mapping[reference_frame]
            )
            self.body_linear_acceleration[reference_frame] = cs.Function(
                f"{body}_linear_acceleration_{reference_frame}",
                [self._q, self._v, self._dv],
                [ddfk(q=self._q, qdot=self._v, qddot=self._dv)["ee_acc_linear"]],
                ["q", "v", "dv"],
                [f"linear_acceleration_{reference_frame}"],
            )

            self.body_angular_acceleration[reference_frame] = cs.Function(
                f"{body}_angular_acceleration_{reference_frame}",
                [self._q, self._v, self._dv],
                [ddfk(q=self._q, qdot=self._v, qddot=self._dv)["ee_acc_angular"]],
                ["q", "v", "dv"],
                [f"angular_acceleration_{reference_frame}"],
            )

            djacobian = self._kindyn.jacobianTimeVariation(
                body_urdf_name, self._frame_mapping[reference_frame]
            )

            self.body_jacobian_derivative[reference_frame] = cs.Function(
                f"{body}_djacobian_{reference_frame}",
                [self._q, self._v],
                [djacobian(q=self._q, v=self._v)["dJ"]],
                ["q", "v"],
                ["djacobian"],
            )

    def update_dynamics(self):
        # Set dynamics of the system and the relative dt

        comfunc = self._kindyn.centerOfMass()
        com_motion = comfunc(q=self._q, v=self._v, a=self._dv)
        self.com = {}
        self.com["position"] = cs.Function(
            "com_position", [self._q], [com_motion["com"]], ["q"], ["com_position"]
        )

        self.com["velocity"] = cs.Function(
            "com_velocity",
            [self._q, self._v],
            [com_motion["vcom"]],
            ["q", "v"],
            ["com_velocity"],
        )

        self.com["acceleration"] = cs.Function(
            "com_acceleration",
            [self._q, self._v, self._dv],
            [com_motion["acom"]],
            ["q", "v", "dv"],
            ["com_acceleration"],
        )

        self.com["jacobian"] = cs.Function(
            "com_jacobian",
            [self._q],
            [self._kindyn.jacobianCenterOfMass(False)(self._q)],
            ["q"],
            ["com_jacobian"],
        )

        jacobian_dt = cs.jacobian(com_motion["acom"], self._v)

        self.com["jacobian_dt"] = cs.Function(
            "com_jacobian_dt",
            [self._q, self._v],
            [jacobian_dt],
            ["q", "v"],
            ["com_jacobian_dt"],
        )

        kinetic_energy = self._kindyn.kineticEnergy()
        potential_energy = self._kindyn.potentialEnergy()

        self.kinetic_energy = cs.Function(
            "kinetic_energy",
            [self._q, self._v],
            [kinetic_energy(self._q, self._v)],
            ["q", "v"],
            ["kinetic_energy"],
        )

        self.potential_energy = cs.Function(
            "potential_energy",
            [self._q],
            [potential_energy(self._q)],
            ["q"],
            ["potential_energy"],
        )

        inertia_matrix = self._kindyn.crba()
        self.inertia_matrix = cs.Function(
            "inertia", [self._q], [inertia_matrix(q=self._q)["B"]], ["q"], ["inertia"]
        )

        ind = self._kindyn.rnea()
        self.rnea = cs.Function(
            "inverse_dynamics",
            [self._q, self._v, self._dv],
            [ind(q=self._q, v=self._v, a=self._dv)["tau"]],
            ["q", "v", "dv"],
            ["tau"],
        )

        fd = self._kindyn.aba()  # this is the forward dynamics function

        self.aba = cs.Function(
            "forward_dynamics",
            [self._q, self._v, self._tau],
            [fd(q=self._q, v=self._v, tau=self._tau)["a"]],
            ["q", "v", "tau"],
            ["dv"],
        )
