from enum import Enum
from dataclasses import dataclass
from typing import Dict
from ..utils.arrays import ArrayLike


class Frame(Enum):
    LOCAL = 1
    WORLD = 2
    LOCAL_WORLD_ALIGNED = 3

    @classmethod
    def from_str(cls, string: str) -> "Frame":
        if string == "local":
            return cls.LOCAL
        elif string == "world":
            return cls.WORLD
        elif string == "world_aligned":
            return cls.LOCAL_WORLD_ALIGNED
        else:
            raise ValueError(f"Unknown frame type: {string}")


class JointType(Enum):
    OMIT = 1  # empty = none = no joint = skip
    FREE_FLYER = 2
    PLANAR = 3

    @classmethod
    def from_str(cls, string: str) -> "JointType":
        if string == "omit":
            return cls.OMIT
        elif string == "free_flyer":
            return cls.FREE_FLYER
        elif string == "planar":
            return cls.PLANAR
        else:
            raise ValueError(f"Unknown joint type: {string}")


@dataclass
class BodyInfo:
    position: ArrayLike
    rotation: ArrayLike
    quaternion: ArrayLike
    jacobian: Dict[Frame, ArrayLike]
    djacobian: Dict[Frame, ArrayLike]
    lin_vel: Dict[Frame, ArrayLike]
    ang_vel: Dict[Frame, ArrayLike]
    lin_acc: Dict[Frame, ArrayLike]
    ang_acc: Dict[Frame, ArrayLike]


@dataclass
class CentroidalDynamics:
    """
    linear: linear momentum
    angular: angular momentum
    linear_dt: linear momentum derivative
    angular_dt: angular momentum derivative
    matrix: centroidal momentum matrix
    matrix_dt: same as linear momentum derivative w.r.t. q
    dynamics_jacobian_q: momentum derivative w.r.t. q
    dynamics_jacobian_v: momentum derivative w.r.t. v
    dynamics_jacobian_vdot: momentum derivative w.r.t. dv

    Under the hood uses pinocchio methods:
        - computeCentroidalMomentumTimeVariation
        - computeCentroidalDynamicsDerivatives
    """

    linear: ArrayLike
    angular: ArrayLike
    linear_dt: ArrayLike
    angular_dt: ArrayLike
    matrix: ArrayLike
    matrix_dt: ArrayLike
    dynamics_jacobian_q: ArrayLike
    dynamics_jacobian_v: ArrayLike
    dynamics_jacobian_dv: ArrayLike
