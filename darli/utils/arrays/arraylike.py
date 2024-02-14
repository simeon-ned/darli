import abc
import numpy.typing as npt


class ArrayLike(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    """This class has to implemented the following operators: """

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def __rsub__(self, other):
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        pass

    @abc.abstractmethod
    def __matmul__(self, other):
        pass

    @abc.abstractmethod
    def __rmatmul__(self, other):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __setitem__(self, key, value):
        pass

    @abc.abstractmethod
    def __truediv__(self, other):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        """
        Returns: Transpose of the array
        """
        pass


class ArrayLikeFactory(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    @abc.abstractmethod
    def zeros(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def eye(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: identity matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def array(*x) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension
        """
        pass

    @abc.abstractmethod
    def solve(self, a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            a (npt.ArrayLike): matrix
            b (npt.ArrayLike): vector

        Returns:
            npt.ArrayLike: solution of the linear system a * x = b
        """
        pass
