from dataclasses import dataclass
from typing import Union
import numpy.typing as npt
from .arraylike import ArrayLike, ArrayLikeFactory
from .numpy import NumpyLike
import casadi as cs


@dataclass
class CasadiLike(ArrayLike):
    """Wrapper class for Casadi types"""

    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        if type(other) in [CasadiLike, NumpyLike]:
            return CasadiLike(self.array @ other.array)
        else:
            return CasadiLike(self.array @ other)

    def __rmatmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        if type(other) in [CasadiLike, NumpyLike]:
            return CasadiLike(other.array @ self.array)
        else:
            return CasadiLike(other @ self.array)

    def __mul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __rmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __add__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __radd__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __sub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __rsub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __neg__(self) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(-self.array)

    def __truediv__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return CasadiLike(self.array / other.array)
        else:
            return CasadiLike(self.array / other)

    def __setitem__(self, idx, value: Union["CasadiLike", npt.ArrayLike]):
        """Overrides set item operator"""
        self.array[idx] = value.array if type(self) is type(value) else value

    def __getitem__(self, idx) -> "CasadiLike":
        """Overrides get item operator"""
        return CasadiLike(self.array[idx])

    @property
    def T(self) -> "CasadiLike":
        """
        Returns:w
            CasadiLike: Transpose of the array
        """
        return CasadiLike(self.array.T)


class CasadiLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x: int) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Matrix of zeros of dim *x
        """
        return CasadiLike(cs.SX.zeros(*x))

    @staticmethod
    def eye(x: int) -> "CasadiLike":
        """
        Args:
            x (int): matrix dimension

        Returns:
            CasadiLike: Identity matrix
        """
        return CasadiLike(cs.SX.eye(x))

    @staticmethod
    def array(*x) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """
        return CasadiLike(cs.SX.sym(*x))

    @staticmethod
    def norm_2(x) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """
        return CasadiLike(cs.norm_2(x))

    @staticmethod
    def solve(A: "CasadiLike", b: "CasadiLike") -> "CasadiLike":
        """
        Args:
            A (CasadiLike): Matrix
            b (CasadiLike): Vector

        Returns:
            CasadiLike: Solution of the linear system
        """
        return CasadiLike(cs.solve(A, b))
