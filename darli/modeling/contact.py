from ..backend import (
    BackendBase,
    Frame,
    CasadiBackend,
    ConeBase,
    CasadiCone,
    PinocchioCone,
)
from .base import ContactBase


class Contact(ContactBase):
    def __init__(self, name: str, backend: BackendBase, frame: Frame, type="point"):
        self.__name = name
        self.__backend: BackendBase = backend
        self.__frame: Frame = frame
        self.__type = type

        if self.__type == "point":
            self.__dim = 3
        elif self.__type == "wrench":
            self.__dim = 6
        else:
            raise ValueError(f"Unknown contact type: {self.__type}")

        # contact jacobian matches the dimension of contact force
        self.__jacobian = (
            self.__backend.update_body(self.__name)
            .jacobian[self.__frame]
            .T[:, : self.dim]
        )
        # print(f"shape of jacobian: {self.__jacobian.shape}")

        if isinstance(self.__backend, CasadiBackend):
            self.__force = self.__backend.math.array(
                f"force_{self.__name}", self.dim
            ).array
        else:
            self.__force = self.__backend.math.zeros(self.dim).array

        self.__contact_qforce = self.__jacobian @ self.__force

        self.__cone = None

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def backend(self) -> BackendBase:
        return self.__backend

    @property
    def name(self):
        return self.__name

    @property
    def ref_frame(self):
        return self.__frame

    def update(self):
        self.__jacobian = (
            self.__backend.update_body(self.__name)
            .jacobian[self.__frame]
            .T[:, : self.dim]
        )

        self.__contact_qforce = self.__jacobian @ self.__force

    @property
    def jacobian(self):
        return self.__jacobian

    @property
    def force(self):
        return self.__force

    @force.setter
    def force(self, value):
        self.__force = value
        self.__contact_qforce = self.__jacobian @ self.__force

        if self.__cone is not None:
            self.__cone.force = self.__force

    @property
    def qforce(self):
        return self.__contact_qforce

    @property
    def cone(self) -> ConeBase:
        if self.__cone is None:
            raise ValueError("Cone constraint is not defined")

        return self.__cone

    def add_cone(self, mu, X=None, Y=None):
        """
        add_cone adds a cone constraint to the contact
        """
        if isinstance(self.__backend, CasadiBackend):
            self.__cone = CasadiCone(self.__force, mu, self.__type, X, Y)
        else:
            self.__cone = PinocchioCone(self.__force, mu, self.__type, X, Y)
