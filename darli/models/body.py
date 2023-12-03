from dataclasses import dataclass
from darli.models.contact import Contact
import casadi as cs
from typing import Dict
import numpy.typing as npt


@dataclass
class Frame:
    local: cs.Function | npt.ArrayLike
    world: cs.Function | npt.ArrayLike
    world_aligned: cs.Function | npt.ArrayLike


class Body:
    """Class to represent body of the model"""

    def __init__(
        self,
        name: Dict[str, str] | str,
        kindyn_backend,
        contact_type=None,
        update=True,
    ):
        # if dictionary then this is map name -> urdf_name
        if isinstance(name, dict):
            self.name = list(name.keys())[0]
            self.urdf_name = name[self.name]
        else:
            self.name = name
            self.urdf_name = name

        self.__backend = kindyn_backend

        self.__position = None
        self.__rotation = None

        self.__contact = None
        self.__contact_type = contact_type
        self.__jacobian = None
        self.__jacobian_dt = None
        self.__linear_velocity = None
        self.__angular_velocity = None
        self.__linear_acceleration = None
        self.__angular_acceleration = None

        if update:
            self.update()

    @property
    def contact(self) -> Contact:
        if self.__contact is None:
            raise ValueError("There is no contact, run `add_contact()` first")

        return self.__contact

    @property
    def position(self):
        if self.__position is None:
            raise ValueError("Position is not calculated, run `update()` first")
        return self.__position

    @property
    def rotation(self):
        if self.__rotation is None:
            raise ValueError("Rotation is not calculated, run `update()` first")
        return self.__rotation

    @property
    def quaternion(self):
        raise NotImplementedError

    @property
    def jacobian(self) -> Frame:
        if self.__jacobian is None:
            raise ValueError("Jacobian is not calculated, run `update()` first")
        return self.__jacobian

    @property
    def jacobian_dt(self) -> Frame:
        if self.__jacobian_dt is None:
            raise ValueError(
                "Jacobian derivative is not calculated, run `update()` first"
            )
        return self.__jacobian_dt

    @property
    def linear_velocity(self) -> Frame:
        if self.__linear_velocity is None:
            raise ValueError("Linear velocity is not calculated, run `update()` first")
        return self.__linear_velocity

    @property
    def angular_velocity(self) -> Frame:
        if self.__angular_velocity is None:
            raise ValueError("Angular velocity is not calculated, run `update()` first")
        return self.__angular_velocity

    @property
    def linear_acceleration(self) -> Frame:
        if self.__linear_acceleration is None:
            raise ValueError(
                "Linear acceleration is not calculated, run `update()` first"
            )
        return self.__linear_acceleration

    @property
    def angular_acceleration(self) -> Frame:
        if self.__angular_acceleration is None:
            raise ValueError(
                "Angular acceleration is not calculated, run `update()` first"
            )
        return self.__angular_acceleration

    def update(self):
        self.__backend.update_body(self.name, self.urdf_name)

        self.__position = self.__backend.body_position
        self.__rotation = self.__backend.body_rotation

        jac = {}
        jac_dt = {}
        lin_vel = {}
        ang_vel = {}
        lin_acc = {}
        ang_acc = {}

        for reference_frame in self.__backend.frame_types:
            jac[reference_frame] = self.__backend.body_jacobian[reference_frame]
            jac_dt[reference_frame] = self.__backend.body_jacobian_derivative[
                reference_frame
            ]
            lin_vel[reference_frame] = self.__backend.body_linear_velocity[
                reference_frame
            ]
            ang_vel[reference_frame] = self.__backend.body_angular_velocity[
                reference_frame
            ]
            lin_acc[reference_frame] = self.__backend.body_linear_acceleration[
                reference_frame
            ]
            ang_acc[reference_frame] = self.__backend.body_angular_acceleration[
                reference_frame
            ]

        self.__jacobian = Frame(
            **jac.copy(),
        )
        self.__linear_velocity = Frame(
            **lin_vel.copy(),
        )
        self.__angular_velocity = Frame(**ang_vel.copy())
        self.__linear_acceleration = Frame(**lin_acc.copy())
        self.__angular_acceleration = Frame(**ang_acc.copy())
        self.__jacobian_dt = Frame(**jac_dt.copy())

        if self.__contact_type is not None:
            self.add_contact(self.__contact_type)

    def add_contact(self, contact_type="point", frame="world_aligned"):
        self.__contact_type = contact_type
        self.__contact = Contact(
            self.name,
            contact_type=self.__contact_type,
            kindyn_backend=self.__backend,
            frame=frame,
        )

    def __str__(self):
        return f"{self.name} Body Object linked to urdf {self.urdf_name}"
