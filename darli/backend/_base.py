from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
from ..utils.arrays import ArrayLike, ArrayLikeFactory
import pinocchio as pin
import numpy as np
import numpy.typing as npt


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


class ConeBase(ABC):
    @abstractmethod
    def full(self, force: ArrayLike | None) -> ArrayLike:
        pass

    @abstractmethod
    def linear(self, force: ArrayLike | None) -> ArrayLike:
        pass


class PinocchioBased:
    def __init__(
        self,
        urdf_path: str,
        root_joint: JointType | None = None,
        fixed_joints: Dict[str, float | npt.ArrayLike] = None,
    ) -> None:
        if fixed_joints is None:
            fixed_joints = {}

        self.__urdf_path = urdf_path

        joint_types = {
            JointType.FREE_FLYER: pin.JointModelFreeFlyer(),
            JointType.PLANAR: pin.JointModelPlanar(),
        }

        # pass root_joint if specified
        if root_joint is None or root_joint == JointType.OMIT:
            model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        else:
            model: pin.Model = pin.buildModelFromUrdf(
                urdf_path, joint_types[root_joint]
            )

        # freeze joints and update coordinate
        freeze_joint_indices = []
        zero_q = pin.neutral(model)
        for joint_name, joint_value in fixed_joints.items():
            # check that this joint_name exists
            if not model.existJointName(joint_name):
                raise ValueError(
                    f"Joint {joint_name} does not exist in the model. Check the spelling"
                )

            joint_id = model.getJointId(joint_name)
            freeze_joint_indices.append(joint_id)

            if not isinstance(joint_value, float):
                zero_q[:7] = joint_value
            else:
                zero_q[joint_id] = joint_value

        self._pinmodel: pin.Model = pin.buildReducedModel(
            model,
            freeze_joint_indices,
            zero_q,
        )
        self._pindata: pin.Data = self._pinmodel.createData()

    @property
    def urdf_path(self) -> str:
        """Returns the path to the URDF file used to build the model."""
        return self.__urdf_path

    @property
    def total_mass(self) -> float:
        """Returns the total mass of the robot."""
        return self._pindata.mass[0]

    @property
    def nq(self) -> int:
        return self._pinmodel.nq

    @property
    def nv(self) -> int:
        return self._pinmodel.nv

    @property
    def nbodies(self) -> int:
        return self._pinmodel.nbodies - 1

    @property
    def q_min(self) -> int:
        return self._pinmodel.lowerPositionLimit

    @q_min.setter
    def q_min(self, value: int):
        self._pinmodel.lowerPositionLimit = value

    @property
    def q_max(self) -> int:
        return self._pinmodel.upperPositionLimit

    @q_max.setter
    def q_max(self, value: int):
        self._pinmodel.upperPositionLimit = value

    @property
    def joint_names(self) -> List[str]:
        return list(self._pinmodel.names)

    def joint_id(self, name: str) -> int:
        return self._pinmodel.getJointId(name)

    def base_parameters(self) -> ArrayLike:
        params = []

        for i in range(len(self._pinmodel.inertias) - 1):
            params.extend(self._pinmodel.inertias[i + 1].toDynamicParameters())

        return np.array(params)


class BackendBase(ABC, PinocchioBased):
    math: ArrayLikeFactory

    # @property
    # @abstractmethod
    # def nq(self) -> int:
    #     pass

    # @property
    # @abstractmethod
    # def nv(self) -> int:
    #     pass

    @abstractmethod
    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def rnea(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def aba(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        tau: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def inertia_matrix(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def kinetic_energy(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def potential_energy(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def jacobian(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def jacobian_dt(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def com_pos(self, q: ArrayLike | None = None) -> ArrayLike:
        pass

    @abstractmethod
    def com_vel(
        self, q: ArrayLike | None = None, v: ArrayLike | None = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def com_acc(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def torque_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def kinetic_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def potential_regressor(
        self,
        q: ArrayLike | None = None,
    ) -> ArrayLike:
        pass

    @abstractmethod
    def momentum_regressor(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
    ):
        pass

    @abstractmethod
    def centroidal_dynamics(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dv: ArrayLike | None = None,
    ) -> CentroidalDynamics:
        pass

    @abstractmethod
    def update_body(self, body: str, body_urdf_name: str = None) -> BodyInfo:
        pass

    @abstractmethod
    def cone(
        self, force: ArrayLike | None, mu: float, type: str, X=None, Y=None
    ) -> ConeBase:
        pass

    @abstractmethod
    def integrate_configuration(
        self,
        q: ArrayLike | None = None,
        v: ArrayLike | None = None,
        dt: float = 1,
    ) -> ArrayLike:
        ...
