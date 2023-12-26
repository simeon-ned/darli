from ..base import ContactBase
from ...backend import BackendBase, Frame
from ..contact import Contact
import casadi as cs


class FunctionalContact(ContactBase):
    def __init__(
        self,
        name: str = None,
        backend: BackendBase = None,
        frame: Frame = None,
        type="point",
        contact=None,
    ):
        if contact is not None:
            self.__contact = contact
        else:
            self.__contact = Contact(name, backend, frame, type)

    @classmethod
    def from_contact(cls, contact: Contact):
        return cls(contact=contact)

    @property
    def dim(self) -> int:
        return self.__contact.dim

    @property
    def backend(self) -> BackendBase:
        return self.__backend

    @property
    def name(self):
        return self.__contact.name

    @property
    def ref_frame(self):
        return self.__contact.ref_frame

    def update(self):
        self.__contact.update()

    @property
    def jacobian(self):
        return cs.Function(
            "jacobian",
            [self.__contact.backend._q],
            [self.__contact.jacobian],
        )

    @property
    def force(self):
        return self.__contact.force

    @force.setter
    def force(self, value):
        self.__contact.force = value

    @property
    def qforce(self):
        return cs.Function(
            "qforce",
            [self.__contact.backend._q, self.force],
            [self.__contact.qforce],
        )

    @property
    def cone(self):
        return self.__contact.cone

    def add_cone(self, mu, X=None, Y=None):
        self.__contact.add_cone(mu, X, Y)
