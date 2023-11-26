from symbotics.utils import RecursiveNamespace
from symbotics.models.contact import Contact


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
        self.jacobian = None
        self.jacobian_dt = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.linear_acceleration = None
        self.angular_acceleration = None

        self.__jacobian = self._model._frames_struct.copy()
        self.__jacobian_dt = self._model._frames_struct.copy()
        self.__linear_velocity = self._model._frames_struct.copy()
        self.__angular_velocity = self._model._frames_struct.copy()
        self.__linear_acceleration = self._model._frames_struct.copy()
        self.__angular_acceleration = self._model._frames_struct.copy()
        if update:
            self.update()
        # self.__linear_acceleration = self._model._frames_struct.copy()
        # self.__angular_acceleration = self._model._frames_struct.copy()

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
        # TODO: ADD ACCELERATION
        self.jacobian = RecursiveNamespace(**self.__jacobian.copy())
        self.linear_velocity = RecursiveNamespace(**self.__linear_velocity.copy())
        self.angular_velocity = RecursiveNamespace(**self.__angular_velocity.copy())
        self.linear_acceleration = RecursiveNamespace(
            **self.__linear_acceleration.copy()
        )
        self.angular_acceleration = RecursiveNamespace(
            **self.__angular_acceleration.copy()
        )
        self.jacobian_dt = RecursiveNamespace(**self.__jacobian_dt.copy())
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
