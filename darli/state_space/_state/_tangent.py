import numpy as np


class Tangent:
    def __init__(self, ndx):
        self.ndx = ndx
        self.data = np.zeros(ndx)

    def __add__(self, other):
        # Overload the addition operator (+)
        if isinstance(other, Tangent):
            # If other is a Tangent, perform element-wise addition
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data + other.data
            return new_tangent
        elif isinstance(other, np.ndarray) and other.shape == (self.ndx,):
            # If other is a numpy array with the same shape, perform element-wise addition
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data + other
            return new_tangent
        else:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        # Overload the subtraction operator (-)
        if isinstance(other, Tangent):
            # If other is a Tangent, perform element-wise subtraction
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data - other.data
            return new_tangent
        elif isinstance(other, np.ndarray) and other.shape == (self.ndx,):
            # If other is a numpy array with the same shape, perform element-wise subtraction
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data - other
            return new_tangent
        else:
            raise TypeError("Unsupported operand type for -")

    def __mul__(self, scalar):
        # Overload the multiplication operator (*)
        if isinstance(scalar, (int, float)):
            # If scalar is a numeric value, perform scalar multiplication
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data * scalar
            return new_tangent
        else:
            raise TypeError("Unsupported operand type for *")

    def __truediv__(self, scalar):
        # Overload the division operator (/)
        if isinstance(scalar, (int, float)):
            # If scalar is a numeric value, perform scalar division
            new_tangent = Tangent(self.ndx)
            new_tangent.data = self.data / scalar
            return new_tangent
        else:
            raise TypeError("Unsupported operand type for /")

    def __repr__(self):
        return f"Tangent(ndx={self.ndx}, data={self.data})"

    def __str__(self):
        return f"Tangent vector of dimension {self.ndx}: {self.data}"
