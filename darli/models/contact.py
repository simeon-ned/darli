import casadi as cs

# TODO:
# Contact should take its Jacobian and argument as input


class Contact:
    """Class to represent body of the model"""

    # TODO: Add Rotation matrix

    def __init__(
        self,
        name=None,
        contact_type="point",
        contact_frame=cs.DM.eye(3),
        kindyn_backend=None,
        frame="world_aligned",
    ):
        self._model = kindyn_backend
        self._name = name
        self.type = contact_type
        if self.type == "wrench":
            self.dim = 6
        else:
            self.dim = 3

        self.reference_frame = frame
        self.contact_frame = contact_frame

        contact_jacobian = self._model.body_jacobian[frame](self._model._q)
        self.jacobian = cs.Function(
            f"{self._name}_contact_jacobian_{frame}",
            [self._model._q],
            [contact_jacobian[: self.dim, :]],
            ["q"],
            ["contact_jacobian"],
        )
        contact_jacobian = self._model.body_jacobian[frame](self._model._q)

        # TODO: probably we should premultiply the force by contact frame rotation
        self._force = cs.SX.sym(f"force_{self._name}", self.dim)
        contact_qforce = cs.mtimes(contact_jacobian[: self.dim, :].T, self._force)

        self.contact_qforce = cs.Function(
            f"{self._name}_qforce_{frame}",
            [self._model._q, self._force],
            [contact_qforce],
            ["q", f"{contact_type}_force"],
            ["generilized_force"],
        )

        self._cone = None

    def __str__(self):
        return f"Contact Object linked to {self._name}"

    def add_cone(self, mu, X=None, Y=None):
        """
        add_cone adds a cone constraint to the contact
        """
        self._cone = ConeConstraint(self._force, mu, self.type, X, Y)

    @property
    def cone(self):
        if self._cone is None:
            raise ValueError("Cone constraint not defined")
        return self._cone


class ConeConstraint:
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

    def linearized(self):
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
