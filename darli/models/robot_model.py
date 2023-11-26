from dataclasses import dataclass
from darli.models.backend import KinDynBackend
from darli.models.state_space import StateSpace
from darli.models.body import Body
import casadi as cs
import numpy as np

# TODO:
# Add mapping between v and v
# Add disable gravity feature


@dataclass
class CoM:
    position: cs.Function
    velocity: cs.Function
    acceleration: cs.Function
    jacobian: cs.Function
    jacobian_dt: cs.Function


@dataclass
class Energy:
    kinetic: cs.Function
    potential: cs.Function


class RobotModel:
    def __init__(
        self,
        urdf_path,
        bodies_names=None,
        contacts=None,
        selector_matrix=None,
        #  kindyn_backend = None,
        calculate=True,
    ):
        """
        Initializes the RobotModel instance.

        Args:
        - urdf_path (str): The path to the URDF file.
        - bodies_names (list): The names of the bodies to be included in the model.
        """

        # Build kindyn instance from URDF file
        # will be replaced with pinochiho-3 later on
        self.urdf_path = urdf_path

        # urdf = open(self.urdf_path, 'r').read()
        self._model = KinDynBackend(self.urdf_path)
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self.nv
        # joint_names = self._model.joint_names
        # joint_names.remove('universe')
        self.joint_names = self._model.joint_names
        self.joint_map = {}
        self.joint_iq = self._model.joint_iq

        self.q_min = self._model.q_min
        self.q_max = self._model.q_max

        # internal casadi variables to generate functions
        self._q = self._model._q  # cs.SX.sym('q', self.nq)
        self._v = self._model._v  # cs.SX.sym('v', self.nv)
        self._dv = self._model._dv  # cs.SX.sym('dv', self.nv)
        self._tau = self._model._tau  # cs.SX.sym('tau', self.nv)

        self.bodies = dict()
        self.add_bodies(bodies_names, calculate=False)

        self.set_selector(selector_matrix, calculate=False)

        for joint_name in self.joint_names:
            self.joint_map[joint_name] = self._model.joint_iq(joint_name)

        self.mass = self._model.mass

        self.forward_dynamics = None
        self.inverse_dynamics = None
        self.gravity = None
        self.inertia = None
        self.coriolis = None
        self.bias_force = None
        self.momentum = None
        self.lagrangian = None
        self.contact_qforce = None
        self.coriolis_matrix = None

        self.contact_names, self.contact_forces = [], []
        self.contacts_map = {}
        self.contacts = {}

        self._com = None
        self._energy = None

        if calculate:
            self.update_model()

    def set_selector(self, selector_matrix=None, passive_joints=None, calculate=True):
        self.selector = np.eye(self.nv)

        if selector_matrix is not None:
            self.selector = selector_matrix
            self.nu = np.shape(self.selector)[1]

        if passive_joints is not None and selector_matrix is None:
            joint_id = []
            self.nu = self.nv - len(passive_joints)
            for joint in passive_joints:
                if isinstance(joint, str):
                    joint_id.append(self.joint_iq(joint))
                if isinstance(joint, int):
                    joint_id.append(joint)
            self.selector = np.delete(self.selector, joint_id, axis=1)

        self._u = cs.SX.sym("u", self.nu)
        self._qfrc_u = cs.mtimes(self.selector, self._u)

        if calculate:
            self.update_model()

    def add_bodies(self, bodies_names, calculate=True):
        """Adds bodies to the model and update the model"""
        # TODO: CHECK IF BODY ALREADY PRESENTED

        # if bodies_names is None:
        # self.bodies_names = bodies_names
        if bodies_names is not None:
            if not isinstance(bodies_names, dict):
                self.bodies_names = set(bodies_names)
                for body_name in self.bodies_names:
                    body = Body(name=body_name, kindyn_backend=self._model)
                    self.bodies[body_name] = body

            else:
                self.bodies_names = bodies_names
                for body_pairs in self.bodies_names.items():
                    # print(dict([body_pairs]))
                    body = Body(name=dict([body_pairs]), kindyn_backend=self._model)

                    self.bodies[body_pairs[0]] = body

    def body(self, body_name) -> Body:
        try:
            return self.bodies[body_name]
        except KeyError:
            raise KeyError(f"Body {body_name} is not added")

    def update_bodies(self):
        for body_name in self.bodies:
            self.body(body_name).update()

    def com(self) -> CoM:
        if self._com is None:
            raise ValueError("CoM is not calculated, run `update_model()` first")
        return self._com

    def energy(self) -> Energy:
        if self._energy is None:
            raise ValueError("Energy is not calculated, run `update_model()` first")
        return self._energy

    def update_model(self):
        if hasattr(self, "bodies_names"):
            self.update_bodies()

        self.update_dynamics()
        self.update_state_space()

    def update_dynamics(self):
        self._com = CoM(
            position=self._model.com["position"],
            velocity=self._model.com["velocity"],
            acceleration=self._model.com["acceleration"],
            jacobian=self._model.com["jacobian"],
            jacobian_dt=self._model.com["jacobian_dt"],
        )
        self._energy = Energy(
            kinetic=self._model.kinetic_energy,
            potential=self._model.potential_energy,
        )

        self.lagrangian = cs.Function(
            "lagrangian",
            [self._q, self._v],
            [
                self._model.kinetic_energy(self._q, self._v)
                - self._model.potential_energy(self._q)
            ],
            ["q", "v"],
            ["lagrangian"],
        )

        self.inertia = self._model.inertia_matrix
        qforce_sum = 0
        for body in self.bodies.values():
            # body = getattr(self, body_name)
            if body.contact is not None:
                qforce = body.contact.contact_qforce(self._q, body.contact._force)
                self.contact_names.append(body.name)
                # print(body.contact._force)
                self.contact_forces.append(body.contact._force)
                # print(self.contact_names, self.contact_forces)
                qforce_sum += qforce

        ind = self._model.rnea
        tau = ind(q=self._q, v=self._v, dv=self._dv)["tau"] - qforce_sum

        self.inverse_dynamics = cs.Function(
            "inverse_dynamics",
            [self._q, self._v, self._dv, *self.contact_forces],
            [tau],
            ["q", "v", "dv", *self.contact_names],
            ["tau"],
        )

        tau_grav = ind(q=self._q, v=np.zeros(self.nv), dv=np.zeros(self.nv))["tau"]

        self.gravity = cs.Function("gravity", [self._q], [tau_grav], ["q"], ["gravity"])

        tau_bias = ind(q=self._q, v=self._v, dv=np.zeros(self.nv))["tau"]

        self.bias_force = cs.Function(
            "bias_force", [self._q, self._v], [tau_bias], ["q", "v"], ["bias_force"]
        )
        coriolis = tau_bias - tau_grav
        self.coriolis = cs.Function(
            "coriolis", [self._q, self._v], [coriolis], ["q", "v"], ["coriolis"]
        )

        coriolis_matrix = cs.jacobian(coriolis, self._v)
        self.coriolis_matrix = cs.Function(
            "coriolis_matrix",
            [self._q, self._v],
            [coriolis_matrix],
            ["q", "v"],
            ["coriolis_matrix"],
        )

        fd = self._model.aba  # this is the forward dynamics function
        dv = fd(q=self._q, v=self._v, tau=self._qfrc_u + qforce_sum)["dv"]

        self.forward_dynamics = cs.Function(
            "forward_dynamics",
            [self._q, self._v, self._u, *self.contact_forces],
            [dv],
            ["q", "v", "u", *self.contact_names],
            ["dv"],
        )

        if len(self.contact_forces) > 0:
            self.contact_qforce = cs.Function(
                "contacts_qforce",
                [self._q, *self.contact_forces],
                [qforce_sum],
                ["q", *self.contact_names],
                ["contacts_qforce"],
            )

    def update_state_space(self):
        self.state_space = StateSpace(model=self, update=True)
