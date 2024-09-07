import numpy as np
from ...backend.liecasadi import SO3, SO3Tangent


# StateSpace.state
# state() =
# groups = [Vector(3), SO(3), Vector(nj)]
#
#
# nx =


class State:
    def __init__(self, nx, ndx, groups):
        self.nx = nx
        self.ndx = ndx
        self.groups = groups
        self.data = np.zeros(nx)
        self.tangent_data = np.zeros(ndx)
        self.time = 0.0
        self.tangent = None

    def log(self):
        # Implement the logarithmic map from the state to the tangent space
        pass

    def exp(self, tangent):
        # Implement the exponential map from the tangent space to the state
        pass

    def tangent(self):
        # Return the tangent vector of the state
        return self.tangent_data

    def time_variation(self, tangent):
        # Compute the time variation of the state given a tangent vector
        pass

    def tangent_map(self, tangent):
        # Map a tangent vector to the tangent space of the state
        pass

    def configuration_space(self):
        # Return the configuration space of the state
        pass

    def state_mapping(self, other_state):
        # Map another state to the same configuration space as the current state
        pass

    def tangent_difference(self, other_state):
        # Compute the tangent difference between two states
        pass

    def random(self):
        # Generate a random state respecting the group geometry
        pass

    def neutral(self):
        # Return the neutral state (identity element) of the configuration space
        pass

    def __add__(self, other):
        # Overload the addition operator (+)
        if isinstance(other, State):
            # If other is a State, perform state addition
            new_state = State(self.nx, self.ndx, self.groups)
            new_state.data = self.data + other.data
            return new_state
        elif isinstance(other, np.ndarray) and other.shape == (self.ndx,):
            # If other is a tangent vector, perform state-tangent addition
            new_state = State(self.nx, self.ndx, self.groups)
            new_state.data = self.data + other
            return new_state
        else:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        # Overload the subtraction operator (-)
        if isinstance(other, State):
            # If other is a State, perform state subtraction
            tangent = self.data - other.data
            return tangent
        else:
            raise TypeError("Unsupported operand type for -")

    def __mul__(self, scalar):
        # Overload the multiplication operator (*)
        if isinstance(scalar, (int, float)):
            # If scalar is a numeric value, perform state-scalar multiplication
            new_state = State(self.nx, self.ndx, self.groups)
            new_state.data = self.data * scalar
            return new_state
        else:
            raise TypeError("Unsupported operand type for *")

    def __truediv__(self, scalar):
        # Overload the division operator (/)
        if isinstance(scalar, (int, float)):
            # If scalar is a numeric value, perform state-scalar division
            new_state = State(self.nx, self.ndx, self.groups)
            new_state.data = self.data / scalar
            return new_state
        else:
            raise TypeError("Unsupported operand type for /")
