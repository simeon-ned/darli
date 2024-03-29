import casadi as cs
from ...backend import BackendBase, Frame
from .._base import BodyBase
from .._body import Body, FrameQuantity
from ._contact import FunctionalContact


class FunctionalBody(BodyBase):
    def __init__(
        self,
        name=None,
        backend: BackendBase = None,
        contact_type=None,
        body=None,
    ):
        if body is not None:
            self.__body = body
        else:
            self.__body = Body(name, backend, contact_type)

    @classmethod
    def from_body(cls, body: Body):
        return cls(body=body)

    @property
    def backend(self) -> BackendBase:
        return self.__body.backend

    @property
    def contact(self):
        return FunctionalContact.from_contact(self.__body.contact)

    @property
    def position(self):
        return cs.Function(
            "position",
            [self.__body.backend._q],
            [self.__body.position],
        )

    @property
    def rotation(self):
        return cs.Function(
            "rotation",
            [self.__body.backend._q],
            [self.__body.rotation],
        )

    @property
    def quaternion(self):
        return cs.Function(
            "quaternion",
            [self.__body.backend._q],
            [self.__body.quaternion],
        )

    @property
    def jacobian(self):
        return FrameQuantity(
            local=cs.Function(
                "jacobian_local",
                [self.__body.backend._q],
                [self.__body.jacobian.local],
            ),
            world=cs.Function(
                "jacobian_world",
                [self.__body.backend._q],
                [self.__body.jacobian.world],
            ),
            world_aligned=cs.Function(
                "jacobian_world_aligned",
                [self.__body.backend._q],
                [self.__body.jacobian.world_aligned],
            ),
        )

    def get_jacobian(self, frame: Frame):
        return cs.Function(
            "jacobian_" + str(frame),
            [self.__body.backend._q],
            [self.__body.get_jacobian[frame]],
        )

    @property
    def jacobian_dt(self):
        return FrameQuantity(
            local=cs.Function(
                "jacobian_dt_local",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.jacobian_dt.local],
            ),
            world=cs.Function(
                "jacobian_dt_world",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.jacobian_dt.world],
            ),
            world_aligned=cs.Function(
                "jacobian_dt_world_aligned",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.jacobian_dt.world_aligned],
            ),
        )

    def get_jacobian_dt(self, frame: Frame):
        return cs.Function(
            "jacobian_dt_" + str(frame),
            [self.__body.backend._q, self.__body.backend._v],
            [self.__body.get_jacobian_dt[frame]],
        )

    @property
    def linear_velocity(self):
        return FrameQuantity(
            local=cs.Function(
                "linear_velocity_local",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.linear_velocity.local],
            ),
            world=cs.Function(
                "linear_velocity_world",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.linear_velocity.world],
            ),
            world_aligned=cs.Function(
                "linear_velocity_world_aligned",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.linear_velocity.world_aligned],
            ),
        )

    def get_linear_velocity(self, frame: Frame):
        return cs.Function(
            "linear_velocity_" + str(frame),
            [self.__body.backend._q, self.__body.backend._v],
            [self.__body.get_linear_velocity[frame]],
        )

    @property
    def angular_velocity(self):
        return FrameQuantity(
            local=cs.Function(
                "angular_velocity_local",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.angular_velocity.local],
            ),
            world=cs.Function(
                "angular_velocity_world",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.angular_velocity.world],
            ),
            world_aligned=cs.Function(
                "angular_velocity_world_aligned",
                [self.__body.backend._q, self.__body.backend._v],
                [self.__body.angular_velocity.world_aligned],
            ),
        )

    def get_angular_velocity(self, frame: Frame):
        return cs.Function(
            "angular_velocity_" + str(frame),
            [self.__body.backend._q, self.__body.backend._v],
            [self.__body.get_angular_velocity[frame]],
        )

    @property
    def linear_acceleration(self):
        return FrameQuantity(
            local=cs.Function(
                "linear_acceleration_local",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.linear_acceleration.local],
            ),
            world=cs.Function(
                "linear_acceleration_world",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.linear_acceleration.world],
            ),
            world_aligned=cs.Function(
                "linear_acceleration_world_aligned",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.linear_acceleration.world_aligned],
            ),
        )

    def get_linear_acceleration(self, frame: Frame):
        return cs.Function(
            "linear_acceleration_" + str(frame),
            [self.__body.backend._q, self.__body.backend._v, self.__body.backend._dv],
            [self.__body.get_linear_acceleration[frame]],
        )

    @property
    def angular_acceleration(self):
        return FrameQuantity(
            local=cs.Function(
                "angular_acceleration_local",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.angular_acceleration.local],
            ),
            world=cs.Function(
                "angular_acceleration_world",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.angular_acceleration.world],
            ),
            world_aligned=cs.Function(
                "angular_acceleration_world_aligned",
                [
                    self.__body.backend._q,
                    self.__body.backend._v,
                    self.__body.backend._dv,
                ],
                [self.__body.angular_acceleration.world_aligned],
            ),
        )

    def get_angular_acceleration(self, frame: Frame):
        return cs.Function(
            "angular_acceleration_" + str(frame),
            [self.__body.backend._q, self.__body.backend._v, self.__body.backend._dv],
            [self.__body.get_angular_acceleration[frame]],
        )

    def update(self):
        self.__body.update()

    def add_contact(self, contact_type="point", frame=Frame.LOCAL_WORLD_ALIGNED):
        self.__body.add_contact(contact_type, frame)
