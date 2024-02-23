from ._common import CommonStateSpace
from ..model._base import ModelBase
import casadi as cs


class PinocchioStateSpace(CommonStateSpace):
    def __init__(self, model: ModelBase) -> None:
        super().__init__(model)

    @property
    def state_jacobian(self):
        # FIXME: separate implementation
        raise NotImplementedError

    @property
    def input_jacobian(self):
        # FIXME: separate implementation
        raise NotImplementedError

    def force_jacobian(self, body_name: str) -> cs.Function:
        # FIXME: separate implementation
        raise NotImplementedError
