from ._common import CommonStateSpace
from ..model._model import Model
from ..model.functional._model import Functional
from ..parametric._functional import Functional as ParametricFunctional
from ..parametric._model import Model as ParametricModel
from ._functional import FunctionalStateSpace
from ..model._base import ModelBase


class StateSpace:
    def __new__(cls, model: ModelBase):
        # Debugging: Print the types to see what's actually being passed
        print(f"Type of model parameter: {type(model)}")
        print(f"Type of Model: {Model}")
        print(f"Type of Functional: {Functional}")

        # Check if model is an instance of Model
        if isinstance(model, Model) | isinstance(model, ParametricModel):
            return CommonStateSpace(model)
        # Check if model is an instance of Functional
        elif isinstance(model, Functional) | isinstance(model, ParametricFunctional):
            return FunctionalStateSpace(model)
        # If neither, raise an error
        else:
            raise ValueError("The model should be either Model or Functional")
