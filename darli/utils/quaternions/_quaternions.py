from ..arrays import ArrayLikeFactory, NumpyLikeFactory


# CBF
# h(q) >= 0
# ddh + beta*dh + alpha*h >= 0
# J dv + Jdot@v + beta*J@v + alpha*h >= 0
#
# Collision
#
# Jdot = (J_now - J_prev)/dt


# Make it scalar last


def expand_map(math: ArrayLikeFactory = NumpyLikeFactory):
    """
    Computes the expand map matrix for 3-dimensional vectors.
    Args:
        math: The math backend to use.
    Returns:
        The matrix which premultiplication adds a zero row before the input vector.
    """
    H = math.zeros((4, 3)).array
    H[1:4, :] = math.eye(3).array

    return H


def hat(vec, math: ArrayLikeFactory = NumpyLikeFactory):
    """
    Computes the hat operator for a given vector.
    Args:
        vec: The input vector.
        math: The math backend to use.
    Returns:
        The hat operator matrix.
    """
    res = math.zeros((3, 3)).array
    res[0, 1] = -vec[2]
    res[0, 2] = vec[1]
    res[1, 0] = vec[2]
    res[1, 2] = -vec[0]
    res[2, 0] = -vec[1]
    res[2, 1] = vec[0]

    return res


def left_mult(quat, math: ArrayLikeFactory = NumpyLikeFactory):
    """
    Compute the left multiplication matrix of a quaternion.
    Parameters:
    quat (array-like): The quaternion to compute the left multiplication matrix for. (Scalar first, vector second)
    math (ArrayLikeFactory, optional): The backend to use for the computation. Defaults to NumpyLikeFactory.
    Returns:
    array-like: The left multiplication matrix.
    """
    s = quat[0]
    v = quat[1:4]

    L = math.zeros((4, 4)).array
    L[0, 0] = s
    L[0, 1:4] = -v.T
    L[1:4, 0] = v
    L[1:4, 1:4] = s * math.eye(3).array + hat(v, math)

    return L


def tangent_map(quat, math: ArrayLikeFactory = NumpyLikeFactory):
    """
    Generate the matrix for quaternion and angular velocity multiplication.
    Parameters:
    q (npt.ArrayLike): The quaternion.
    Returns:
    np.ndarray: The matrix for quaternion and angular velocity multiplication.
    """

    return left_mult(quat, math) @ expand_map(math)


def state_tangent_map(state, math: ArrayLikeFactory = NumpyLikeFactory):
    """
    Construct the state transition matrix for proper state space linearization
    Args:
        state (npt.ArrayLike): The current state of the system.
    Returns:
        np.ndarray: A (nq x nv) state transition matrix.
    """
    q = state[3:7]  # Orientation (quaternion)

    res = math.zeros((state.shape[0], state.shape[0] - 1)).array
    res[0:3, 0:3] = math.eye(3).array
    res[3:7, 3:6] = tangent_map(q, math)
    res[7:, 6:] = math.eye(state.shape[0] - 7).array

    return res
