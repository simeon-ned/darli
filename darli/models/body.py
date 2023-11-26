from dataclasses import dataclass
from darli.models.contact import Contact
import casadi as cs


@dataclass
class Frame:
    local: cs.Function
    world: cs.Function
    world_aligned: cs.Function


class Body:
    """Class to represent body of the model"""

    def __init__(self, name=None, kindyn_backend=None, contact_type=None, update=True):
        # if dictionary then this is map name -> urdf_name
        if isinstance(name, dict):
            self.name = list(name.keys())[0]
            self.urdf_name = name[self.name]
        else:
            self.name = name
            self.urdf_name = name

        # mappings
        if kindyn_backend is None:
            print("Provide the modeling backend")
            # TODO: Exit
        else:
            self._model = kindyn_backend

        self.position = None
        self.quaternion = None
        self.rotation = None
        self.energy = {"kinetic": None, "potential": None}

        self.contact = None
        self.__contact_type = contact_type
        self._jacobian = None
        self._jacobian_dt = None
        self._linear_velocity = None
        self._angular_velocity = None
        self._linear_acceleration = None
        self._angular_acceleration = None

        self.__jacobian = {}
        self.__jacobian_dt = {}
        self.__linear_velocity = {}
        self.__angular_velocity = {}
        self.__linear_acceleration = {}
        self.__angular_acceleration = {}
        if update:
            self.update()
        # self.__linear_acceleration = self._model._frames_struct.copy()
        # self.__angular_acceleration = self._model._frames_struct.copy()

    def jacobian(self) -> Frame:
        if self._jacobian is None:
            raise ValueError("Jacobian is not calculated, run `update()` first")
        return self._jacobian

    def jacobian_dt(self) -> Frame:
        if self._jacobian_dt is None:
            raise ValueError(
                "Jacobian derivative is not calculated, run `update()` first"
            )
        return self._jacobian_dt

    def linear_velocity(self) -> Frame:
        if self._linear_velocity is None:
            raise ValueError("Linear velocity is not calculated, run `update()` first")
        return self._linear_velocity

    def angular_velocity(self) -> Frame:
        if self._angular_velocity is None:
            raise ValueError("Angular velocity is not calculated, run `update()` first")
        return self._angular_velocity

    def linear_acceleration(self) -> Frame:
        if self._linear_acceleration is None:
            raise ValueError(
                "Linear acceleration is not calculated, run `update()` first"
            )
        return self._linear_acceleration

    def angular_acceleration(self) -> Frame:
        if self._angular_acceleration is None:
            raise ValueError(
                "Angular acceleration is not calculated, run `update()` first"
            )
        return self._angular_acceleration

    def update(self):
        self._model.update_body(self.name, self.urdf_name)

        self.position = self._model.body_position
        self.rotation = self._model.body_rotation
        self.quaternion = None

        for reference_frame in self._model.frame_types:
            self.__jacobian[reference_frame] = self._model.body_jacobian[
                reference_frame
            ]
            self.__jacobian_dt[reference_frame] = self._model.body_jacobian_derivative[
                reference_frame
            ]
            self.__linear_velocity[reference_frame] = self._model.body_linear_velocity[
                reference_frame
            ]
            self.__angular_velocity[
                reference_frame
            ] = self._model.body_angular_velocity[reference_frame]
            self.__linear_acceleration[
                reference_frame
            ] = self._model.body_linear_acceleration[reference_frame]
            self.__angular_acceleration[
                reference_frame
            ] = self._model.body_angular_acceleration[reference_frame]

        self._jacobian = Frame(
            **self.__jacobian.copy(),
        )
        self._linear_velocity = Frame(
            **self.__linear_velocity.copy(),
        )
        self._angular_velocity = Frame(**self.__angular_velocity.copy())
        self._linear_acceleration = Frame(**self.__linear_acceleration.copy())
        self._angular_acceleration = Frame(**self.__angular_acceleration.copy())
        self._jacobian_dt = Frame(**self.__jacobian_dt.copy())
        if self.__contact_type is not None:
            self.add_contact(self.__contact_type)
        else:
            self.contact = None

    def add_contact(self, contact_type="point", frame="world_aligned"):
        self.__contact_type = contact_type
        self.contact = Contact(
            self.name,
            contact_type=self.__contact_type,
            kindyn_backend=self._model,
            frame=frame,
        )

    def __str__(self):
        return f"{self.name} Body Object linked to urdf {self.urdf_name}"
