from dataclasses import dataclass
from ..arrays import ArrayLike
from ..backend import BackendBase, BodyInfo, Frame
from .contact import Contact
from .base import BodyBase
import casadi as cs


@dataclass
class FrameQuantity:
    local: ArrayLike
    world: ArrayLike
    world_aligned: ArrayLike


class Body(BodyBase):
    def __init__(self, name, backend: BackendBase, contact_type=None):
        if isinstance(name, dict):
            self.name = list(name.keys())[0]
            self.urdf_name = name[self.name]
        else:
            self.name = name
            self.urdf_name = name

        self.__backend: BackendBase = backend

        self.__info: BodyInfo | None = None
        self.__contact_type = contact_type
        self.__contact: Contact = None

        self.update()

    @property
    def backend(self) -> BackendBase:
        return self.__backend

    @property
    def contact(self) -> Contact:
        return self.__contact

    @property
    def position(self):
        if self.__info is None:
            raise ValueError("Position is not calculated, run `update()` first")
        return self.__info.position

    @property
    def rotation(self):
        if self.__info is None:
            raise ValueError("Rotation is not calculated, run `update()` first")
        return self.__info.rotation

    @property
    def quaternion(self):
        raise NotImplementedError

    @property
    def jacobian(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Jacobian is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.jacobian[Frame.LOCAL],
            world=self.__info.jacobian[Frame.WORLD],
            world_aligned=self.__info.jacobian[Frame.LOCAL_WORLD_ALIGNED],
        )

    @property
    def jacobian_dt(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError(
                "Jacobian derivative is not calculated, run `update()` first"
            )
        return FrameQuantity(
            local=self.__info.djacobian[Frame.LOCAL],
            world=self.__info.djacobian[Frame.WORLD],
            world_aligned=self.__info.djacobian[Frame.LOCAL_WORLD_ALIGNED],
        )

    @property
    def linear_velocity(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Linear velocity is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.lin_vel[Frame.LOCAL],
            world=self.__info.lin_vel[Frame.WORLD],
            world_aligned=self.__info.lin_vel[Frame.LOCAL_WORLD_ALIGNED],
        )

    @property
    def angular_velocity(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Angular velocity is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.ang_vel[Frame.LOCAL],
            world=self.__info.ang_vel[Frame.WORLD],
            world_aligned=self.__info.ang_vel[Frame.LOCAL_WORLD_ALIGNED],
        )

    @property
    def linear_acceleration(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError(
                "Linear acceleration is not calculated, run `update()` first"
            )
        return FrameQuantity(
            local=self.__info.lin_acc[Frame.LOCAL],
            world=self.__info.lin_acc[Frame.WORLD],
            world_aligned=self.__info.lin_acc[Frame.LOCAL_WORLD_ALIGNED],
        )

    @property
    def angular_acceleration(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError(
                "Angular acceleration is not calculated, run `update()` first"
            )
        return FrameQuantity(
            local=self.__info.ang_acc[Frame.LOCAL],
            world=self.__info.ang_acc[Frame.WORLD],
            world_aligned=self.__info.ang_acc[Frame.LOCAL_WORLD_ALIGNED],
        )

    def update(self):
        self.__info = self.__backend.update_body(self.name, self.urdf_name)

        if self.__contact_type is not None:
            self.add_contact(self.__contact_type)

    def add_contact(self, contact_type="point", frame=Frame.LOCAL_WORLD_ALIGNED):
        self.__contact_type = contact_type
        self.__contact = Contact(
            self.urdf_name,
            self.__backend,
            frame=frame,
            type=contact_type,
        )


class SymbolicBody(BodyBase):
    def __init__(self, name, backend: BackendBase, contact_type=None):
        self.__body = Body(name, backend, contact_type)

    @property
    def backend(self) -> BackendBase:
        return self.__body.backend

    @property
    def contact(self):
        return self.__body.contact

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

    def update(self):
        self.__body.update()

    def add_contact(self, contact_type="point", frame=Frame.LOCAL_WORLD_ALIGNED):
        self.__body.add_contact(contact_type, frame)
