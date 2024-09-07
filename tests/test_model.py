import pytest
import numpy as np
from darli.model import Model
from darli.model.functional import Functional
from darli.backend import PinocchioBackend, CasadiBackend, JointType

# Import model paths
from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
from robot_descriptions.panda_mj_description import MJCF_PATH as PANDA_MJCF_PATH
from robot_descriptions.h1_description import URDF_PATH as H1_URDF_PATH


# Fixture for test data (numeric)
@pytest.fixture
def test_data():
    return {
        "q": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0]),
        "v": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "dv": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        "tau": np.array([0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0, 0]),
    }


@pytest.fixture
def test_data_h1():
    return {
        "q": np.concatenate([np.array([0, 0, 0, 0, 0, 0, 1]), np.zeros(19)]),
        "v": np.zeros(25),
        "dv": np.zeros(25),
        "tau": np.zeros(25),
    }


# Fixture to create Model instance with PinocchioBackend (Panda URDF)
@pytest.fixture
def model_instance_panda_urdf(test_data):
    backend = PinocchioBackend(PANDA_URDF_PATH)
    model = Model(backend)
    return model, test_data


# Fixture to create FunctionalModel instance with CasadiBackend (Panda URDF)
@pytest.fixture
def functional_model_instance_panda_urdf(test_data):
    backend = CasadiBackend(PANDA_URDF_PATH)
    functional_model = Functional(backend)
    return functional_model, test_data


# Fixture to create Model instance with PinocchioBackend (Panda MJCF)
@pytest.fixture
def model_instance_panda_mjcf(test_data):
    backend = PinocchioBackend(PANDA_MJCF_PATH)
    model = Model(backend)
    return model, test_data


# Fixture to create FunctionalModel instance with CasadiBackend (Panda MJCF)
@pytest.fixture
def functional_model_instance_panda_mjcf(test_data):
    backend = CasadiBackend(PANDA_MJCF_PATH)
    functional_model = Functional(backend)
    return functional_model, test_data


# Fixture to create Model instance with PinocchioBackend (H1 URDF)
@pytest.fixture
def model_instance_h1(test_data_h1):
    backend = PinocchioBackend(
        H1_URDF_PATH, root_joint=JointType.FREE_FLYER
    )  # Assuming the H1 URDF requires a free flyer joint
    model = Model(backend)
    return model, test_data_h1


# Fixture to create FunctionalModel instance with CasadiBackend (H1 URDF)
@pytest.fixture
def functional_model_instance_h1(test_data_h1):
    backend = CasadiBackend(H1_URDF_PATH, root_joint=JointType.FREE_FLYER)
    functional_model = Functional(backend)
    return functional_model, test_data_h1


# Helper function to run comparison tests between Model and FunctionalModel
def compare_model_outputs(model, functional_model, data):
    q, v, dv, tau = data["q"], data["v"], data["dv"], data["tau"]

    # compare com
    assert np.allclose(
        model.com(q).position, np.array(functional_model.com.position(q)).flatten()
    )
    assert np.allclose(
        model.com(q, v).velocity,
        np.array(functional_model.com.velocity(q, v)).flatten(),
    )
    assert np.allclose(
        model.com(q, v, dv).acceleration,
        np.array(functional_model.com.acceleration(q, v, dv)).flatten(),
    )
    assert np.allclose(
        model.com(q).jacobian, np.array(functional_model.com.jacobian(q))
    )

    # compare energy
    assert np.isclose(model.energy(q, v).kinetic, functional_model.energy.kinetic(q, v))
    assert np.isclose(model.energy(q).potential, functional_model.energy.potential(q))

    # compare inertia
    assert np.allclose(model.inertia(q), np.array(functional_model.inertia(q)))

    # compare coriolis
    assert np.allclose(
        model.coriolis(q, v), np.array(functional_model.coriolis(q, v)).flatten()
    )

    # compare bias force
    assert np.allclose(
        model.bias_force(q, v), np.array(functional_model.bias_force(q, v)).flatten()
    )

    # compare forward dynamics
    assert np.allclose(
        model.forward_dynamics(q, v, tau),
        np.array(functional_model.forward_dynamics(q, v, tau)).flatten(),
    )

    # compare inverse dynamics
    assert np.allclose(
        model.inverse_dynamics(q, v, dv),
        np.array(functional_model.inverse_dynamics(q, v, dv)).flatten(),
    )


# Test to compare Model and FunctionalModel outputs using the same numeric data
@pytest.mark.parametrize(
    "model_fixture, functional_fixture",
    [
        ("model_instance_panda_urdf", "functional_model_instance_panda_urdf"),
        ("model_instance_panda_mjcf", "functional_model_instance_panda_mjcf"),
        ("model_instance_h1", "functional_model_instance_h1"),
    ],
)
def test_compare_model_functional_model_outputs(
    request, model_fixture, functional_fixture
):
    model, data = request.getfixturevalue(model_fixture)
    functional_model, _ = request.getfixturevalue(functional_fixture)
    compare_model_outputs(model, functional_model, data)
