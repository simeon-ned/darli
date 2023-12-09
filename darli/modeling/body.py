from dataclasses import dataclass
from ..arrays import ArrayLike
from ..backend import BackendBase, BodyInfo, Frame


@dataclass
class FrameQuantity:
    local: ArrayLike
    world: ArrayLike
    world_aligned: ArrayLike


class Body:
    def __init__(self, name, backend: BackendBase, contact_type=None):
        if isinstance(name, dict):
            self.name = list(name.keys())[0]
            self.urdf_name = name[self.name]
        else:
            self.name = name
            self.urdf_name = name

        self.__backend: BackendBase = backend

        self.__info: BodyInfo | None = None

        self.update()

    @property
    def contact(self):
        if self.__info is None:
            raise ValueError("There is no contact, run `add_contact()` first")

        # return
        raise NotImplementedError

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
