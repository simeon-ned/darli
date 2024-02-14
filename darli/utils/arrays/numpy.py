from dataclasses import dataclass
from .arraylike import ArrayLike, ArrayLikeFactory
import numpy.typing as npt
from typing import Union
import numpy as np


@dataclass
class NumpyLike(ArrayLike):
    """Class wrapping NumPy types"""

    array: np.ndarray

    def __setitem__(self, idx, value: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides set item operator"""
        if type(self) is type(value):
            self.array[idx] = value.array.reshape(self.array[idx].shape)
        else:
            self.array[idx] = value

    def __getitem__(self, idx) -> "NumpyLike":
        """Overrides get item operator"""
        return NumpyLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self) -> "NumpyLike":
        """
        Returns:
            NumpyLike: transpose of the array
        """
        return NumpyLike(self.array.T)

    def __matmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return NumpyLike(self.array @ other.array)
        else:
            return NumpyLike(self.array @ np.array(other))

    def __rmatmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return NumpyLike(other.array @ self.array)
        else:
            return NumpyLike(other @ self.array)

    def __mul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return NumpyLike(self.array * other.array)
        else:
            return NumpyLike(self.array * other)

    def __rmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return NumpyLike(other.array * self.array)
        else:
            return NumpyLike(other * self.array)

    def __truediv__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return NumpyLike(self.array / other.array)
        else:
            return NumpyLike(self.array / other)

    def __add__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() + other.squeeze())
        return NumpyLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array + other)
        return NumpyLike(self.array + other.array)

    def __sub__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides - operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() - other.squeeze())
        return NumpyLike(self.array.squeeze() - other.array.squeeze())

    def __rsub__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides - operator"""
        if type(self) is not type(other):
            return NumpyLike(other.squeeze() - self.array.squeeze())
        return NumpyLike(other.array.squeeze() - self.array.squeeze())

    def __neg__(self):
        """Overrides - operator"""
        return NumpyLike(-self.array)


class NumpyLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x) -> "NumpyLike":
        """
        Returns:
            NumpyLike: zero matrix of dimension x
        """
        return NumpyLike(np.zeros(x))

    @staticmethod
    def eye(x: int) -> "NumpyLike":
        """
        Args:
            x (int): matrix dimension

        Returns:
            NumpyLike: Identity matrix of dimension x
        """
        return NumpyLike(np.eye(x))

    @staticmethod
    def array(x) -> "NumpyLike":
        """
        Returns:
            NumpyLike: Vector wrapping *x
        """
        return NumpyLike(np.array(x))

    @staticmethod
    def norm_2(x) -> "NumpyLike":
        """
        Returns:
            NumpyLike: Norm of x
        """
        return NumpyLike(np.linalg.norm(x))

    @staticmethod
    def solve(a: "NumpyLike", b: "NumpyLike") -> "NumpyLike":
        """
        Args:
            a (NumpyLike): Matrix
            b (NumpyLike): Vector

        Returns:
            NumpyLike: Solution of the linear system a @ x = b
        """
        return NumpyLike(np.linalg.solve(a, b))
