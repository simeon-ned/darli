from dataclasses import dataclass
from ..utils.arrays import ArrayLike
from ..backend import BackendBase, BodyInfo, Frame
from ._contact import Contact
from ._base import BodyBase, FrameQuantity


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
        if self.__info is None:
            raise ValueError("Quaternion is not calculated, run `update()` first")
        return self.__info.quaternion

    @property
    def jacobian(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Jacobian is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.jacobian[Frame.LOCAL],
            world=self.__info.jacobian[Frame.WORLD],
            world_aligned=self.__info.jacobian[Frame.LOCAL_WORLD_ALIGNED],
        )

    def get_jacobian(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError("Jacobian is not calculated, run `update()` first")
        return self.__info.jacobian[frame]

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

    def get_jacobian_dt(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError(
                "Jacobian derivative is not calculated, run `update()` first"
            )
        return self.__info.djacobian[frame]

    @property
    def linear_velocity(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Linear velocity is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.lin_vel[Frame.LOCAL],
            world=self.__info.lin_vel[Frame.WORLD],
            world_aligned=self.__info.lin_vel[Frame.LOCAL_WORLD_ALIGNED],
        )

    def get_linear_velocity(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError("Linear velocity is not calculated, run `update()` first")
        return self.__info.lin_vel[frame]

    @property
    def angular_velocity(self) -> FrameQuantity:
        if self.__info is None:
            raise ValueError("Angular velocity is not calculated, run `update()` first")
        return FrameQuantity(
            local=self.__info.ang_vel[Frame.LOCAL],
            world=self.__info.ang_vel[Frame.WORLD],
            world_aligned=self.__info.ang_vel[Frame.LOCAL_WORLD_ALIGNED],
        )

    def get_angular_velocity(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError("Angular velocity is not calculated, run `update()` first")
        return self.__info.ang_vel[frame]

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

    def get_linear_acceleration(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError(
                "Linear acceleration is not calculated, run `update()` first"
            )
        return self.__info.lin_acc[frame]

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

    def get_angular_acceleration(self, frame: Frame) -> ArrayLike:
        if self.__info is None:
            raise ValueError(
                "Angular acceleration is not calculated, run `update()` first"
            )
        return self.__info.ang_acc[frame]

    def update(self):
        self.__info = self.__backend.update_body(self.name, self.urdf_name)

        # update contact
        if self.__contact is not None:
            self.__contact.update(self.__info.jacobian[self.__contact.ref_frame])

        # should run at first occasion, if contact should be added,
        # it will automatically load info about body's jacobian
        if self.__contact_type is not None and self.__contact is None:
            self.add_contact(self.__contact_type)

    def add_contact(
        self, contact_type="point", frame=Frame.LOCAL_WORLD_ALIGNED, constructor=Contact
    ):
        self.__contact_type = contact_type
        self.__contact = constructor(
            self.urdf_name,
            self.__backend,
            frame=frame,
            type=contact_type,
        )
