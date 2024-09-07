import numpy as np


# Sources for inspiration:
# https://github.com/pymanopt/pymanopt/blob/master/src/pymanopt/manifolds/positive_definite.py
# https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/spatial_2inertia_8hpp_source.html

# TODO:
# Check if we can use the cassadie backend:
# def expm(A):
#     """Matrix exponential for a 4x4 matrix using CasADi."""
#     # Compute the matrix exponential using CasADi's expm() function
#     expmA = cs.expm(A)
#
#     return expmA
#
# def logm_pd(A):
#     """Matrix logarithm for a real positive definite matrix using CasADi."""
#     # Compute the Cholesky decomposition of the matrix
#     L = cs.chol(A)
#     # Compute the logarithm of the Cholesky factor
#     L_log = cs.solve(L, cs.eye(4)) * cs.log(cs.diag(L))
#     # Reconstruct the matrix logarithm using the Cholesky factor logarithm
#     logmA = L_log @ L_log.T
#     return logmA
# Maybe we should move this to the <<math>>?


class InertialParameters:
    def __init__(self):
        self.dim = 4
        self.ndx = 10
        self.vector = np.zeros(self.ndx)
        self.matrix = np.eye(self.ndx)
        self.principal_frame = None
        self.principle_inertia = None

    def __add__(self, other):
        result = InertialParameters()
        result.vector = self.vector + other.vector
        result.matrix = self.matrix + other.matrix
        return result

    def __sub__(self, other):
        result = InertialParameters()
        result.vector = self.vector - other.vector
        result.matrix = self.matrix - other.matrix
        return result

    def exp(self, tangent_vector, point=None):
        # Exponential map
        if point is None:
            point = self.matrix

        p_inv_tv = np.linalg.solve(point, tangent_vector)
        w, v = np.linalg.eigh(p_inv_tv)
        # Exponentiate the eigenvalues
        w_exp = np.exp(w)
        # Reconstruct the matrix exponential using the eigen decomposition
        expm_inv = v @ np.diag(w_exp) @ v.T

        return point @ expm_inv

    def log(self, point_a, point_b):
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)

        w, v = np.linalg.eigh(c_inv @ point_b @ c_inv.T)
        # Compute the logarithm of the eigenvalues
        w_log = np.log(w)
        # Reconstruct the matrix logarithm using the eigen decomposition
        logm = v @ np.diag(w_log) @ v.T

        return c @ logm @ c.T

    def retraction(self, tangent_vector):
        # Retraction operation
        point = self.matrix
        p_inv_tv = np.linalg.solve(point, tangent_vector)
        return (point + tangent_vector + tangent_vector @ p_inv_tv / 2) / np.trace(
            point + tangent_vector
        )

    def tangent(self, vector):
        # Tangent space computation
        return np.hstack((self.vector, vector))

    def reimann_jacobian(self, tangent_vector):
        # Riemannian operations
        point = self.matrix
        p_inv = np.linalg.inv(point)
        return p_inv @ tangent_vector @ p_inv

    def projection(self, vector):
        # Projection operation
        return (vector + vector.T) / 2

    def euclidian_jacobian(self, tangent_vector):
        # Euclidean operations
        return tangent_vector


# The following class is used for the collection of InertialParameters
# i.e Lumped parameters
class LumpedParameters(InertialParameters):
    def __init__(self, nb):
        super().__init__()
        # Add any additional attributes specific to LumpedParameters


# The following will be the collection of inertial parameters in reduced form
# i.e. if we have some linear dependency or zero columns in associated regressor
class ReducedParameters(InertialParameters):
    def __init__(self, nb, basis):
        super().__init__()
        # Add any additional attributes specific to ReducedParameters
