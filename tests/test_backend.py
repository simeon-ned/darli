import pytest
import numpy as np
import casadi as cs
from darli.backend import PinocchioBackend, BackendBase, JointType, CasadiBackend

# Import model paths
from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
from robot_descriptions.panda_mj_description import MJCF_PATH as PANDA_MJCF_PATH
from robot_descriptions.h1_description import URDF_PATH as H1_URDF_PATH


# Fixture for test data for the Panda URDF model
@pytest.fixture
def panda_urdf_data():
    return {
        "q": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0]),
        "v": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "dv": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "tau": np.array([0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0, 0]),
        "body_name": "panda_link7",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "point",
    }


# Fixture for test data for the Panda MJCF model
@pytest.fixture
def panda_mjcf_data():
    return {
        "q": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0]),
        "v": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "dv": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "tau": np.array([0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0, 0]),
        "body_name": "link7",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "point",
    }


# Fixture for test data for the H1 URDF model
@pytest.fixture
def h1_urdf_data():
    return {
        "q": np.concatenate([np.array([0, 0, 0, 0, 0, 0, 1]), np.zeros(19)]),
        "v": np.zeros(25),
        "dv": np.zeros(25),
        "tau": np.zeros(25),
        "body_name": "right_elbow_link",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "sliding",
        "dt": 0.02,
    }


# Fixture for test data for the Panda URDF model with Casadi backend
@pytest.fixture
def panda_urdf_data_casadi():
    return {
        "q": cs.SX.sym("q", 9),
        "v": cs.SX.sym("v", 9),
        "dv": cs.SX.sym("dv", 9),
        "tau": cs.SX.sym("tau", 9),
        "body_name": "panda_link7",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "point",
    }


# Fixture for test data for the Panda MJCF model with Casadi backend
@pytest.fixture
def panda_mjcf_data_casadi():
    return {
        "q": cs.SX.sym("q", 9),
        "v": cs.SX.sym("v", 9),
        "dv": cs.SX.sym("dv", 9),
        "tau": cs.SX.sym("tau", 9),
        "body_name": "link7",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "point",
    }


# Fixture for test data for the H1 URDF model with Casadi backend
@pytest.fixture
def h1_urdf_data_casadi():
    return {
        "q": cs.SX.sym("q", 26),
        "v": cs.SX.sym("v", 25),
        "dv": cs.SX.sym("dv", 25),
        "tau": cs.SX.sym("tau", 25),
        "body_name": "right_elbow_link",
        "force": np.array([-9.81, 0.0, 0.0]),
        "mu": 0.4,
        "type": "sliding",
    }


# Helper function to run common tests on any backend
def run_backend_tests(backend: BackendBase, data):
    backend.update(data["q"], data["v"], data["dv"], data["tau"])

    assert backend.rnea(data["q"], data["v"], data["dv"]) is not None
    assert backend.aba(data["q"], data["v"], data["tau"]) is not None
    assert backend.inertia_matrix(data["q"]) is not None
    assert backend.kinetic_energy(data["q"], data["v"]) is not None
    assert backend.potential_energy(data["q"]) is not None
    assert backend.jacobian(data["q"]) is not None
    assert backend.jacobian_dt(data["q"], data["v"]) is not None
    assert backend.com_pos(data["q"]) is not None
    assert backend.com_vel(data["q"], data["v"]) is not None
    assert backend.com_acc(data["q"], data["v"], data["dv"]) is not None
    assert backend.torque_regressor(data["q"], data["v"], data["dv"]) is not None
    assert backend.kinetic_regressor(data["q"], data["v"]) is not None
    assert backend.potential_regressor(data["q"]) is not None

    # test momentum regressor only for PinocchioBackend
    # TODO: change when CasadiBackend implements momentum_regressor
    if isinstance(backend, PinocchioBackend):
        assert backend.momentum_regressor(data["q"], data["v"]) is not None
    assert backend.centroidal_dynamics(data["q"], data["v"], data["dv"]) is not None

    # try update_body with existing and non-existing body names
    assert backend.update_body(data["body_name"]) is not None
    # should return a KeyError
    pytest.raises(KeyError, backend.update_body, "non_existent_body")

    assert backend.cone(data["force"], data["mu"], data["type"]) is not None
    assert backend.integrate_configuration(data["q"], data["v"]) is not None


# Fixture to create backend for the Panda URDF model
@pytest.fixture
def panda_urdf_backend(panda_urdf_data):
    backend_instance = PinocchioBackend(PANDA_URDF_PATH)
    return backend_instance, panda_urdf_data


# Fixture to create backend for the Panda MJCF model
@pytest.fixture
def panda_mjcf_backend(panda_mjcf_data):
    backend_instance = PinocchioBackend(PANDA_MJCF_PATH)
    return backend_instance, panda_mjcf_data


# Fixture to create backend for the H1 URDF model
@pytest.fixture
def h1_urdf_backend(h1_urdf_data):
    backend_instance = PinocchioBackend(H1_URDF_PATH, root_joint=JointType.FREE_FLYER)
    return backend_instance, h1_urdf_data


# Tests for Panda URDF backend
def test_panda_urdf_backend(panda_urdf_backend):
    backend, data = panda_urdf_backend
    run_backend_tests(backend, data)


# Tests for Panda MJCF backend
def test_panda_mjcf_backend(panda_mjcf_backend):
    backend, data = panda_mjcf_backend
    run_backend_tests(backend, data)


# Tests for H1 URDF backend
def test_h1_urdf_backend(h1_urdf_backend):
    backend, data = h1_urdf_backend
    run_backend_tests(backend, data)


# Fixture to create Casadi backend for the Panda URDF model
@pytest.fixture
def panda_urdf_backend_casadi(panda_urdf_data_casadi):
    backend_instance = CasadiBackend(PANDA_URDF_PATH)
    return backend_instance, panda_urdf_data_casadi


# Fixture to create Casadi backend for the Panda MJCF model
@pytest.fixture
def panda_mjcf_backend_casadi(panda_mjcf_data_casadi):
    backend_instance = CasadiBackend(PANDA_MJCF_PATH)
    return backend_instance, panda_mjcf_data_casadi


# Fixture to create Casadi backend for the H1 URDF model
@pytest.fixture
def h1_urdf_backend_casadi(h1_urdf_data_casadi):
    backend_instance = CasadiBackend(H1_URDF_PATH, root_joint=JointType.FREE_FLYER)
    return backend_instance, h1_urdf_data_casadi


# Tests for Casadi backend with Panda URDF model
def test_panda_urdf_backend_casadi(panda_urdf_backend_casadi):
    backend, data = panda_urdf_backend_casadi
    run_backend_tests(backend, data)


# Tests for Casadi backend with Panda MJCF model
def test_panda_mjcf_backend_casadi(panda_mjcf_backend_casadi):
    backend, data = panda_mjcf_backend_casadi
    run_backend_tests(backend, data)


# Tests for Casadi backend with H1 URDF model
def test_h1_urdf_backend_casadi(h1_urdf_backend_casadi):
    backend, data = h1_urdf_backend_casadi
    run_backend_tests(backend, data)
