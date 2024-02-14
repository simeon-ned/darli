# DARLi: Differentiable Articulated Robotics Library

The DARLi (Differentiable Articulated Robotics Library) is a wrapper around the CasADi kinematics/dynamics and Pinocchio libraries. Its primary goal is to facilitate the creation of differentiable models for robotic systems, provided a URDF (Unified Robot Description Format) file. **DARLi is not an implementation of mechanics-oriented** algorithms, such as Featherstone's Articulated Body or the Recursive Newton-Euler. Instead, we rely on the efficient implementations provided by [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/master) and offer a wrapper that grants easy access to a suite of features that we and our colleagues have found useful in practice of trajectory planning, feedback control, and system identification over poly-articulated robots.

### Features of DARLi:

- **Different Backends**: Currently, DARLi supports the numerical Pinocchio backend and its CasADi wrapper, based on a [modified version](https://github.com/lvjonok/casadi_kin_dyn) of [casadi_kin_dyn](https://github.com/ADVRHumanoids/casadi_kin_dyn)  (will be replaced with [Pinocchio 3](https://github.com/stack-of-tasks/pinocchio/tree/pinocchio3-preview) upon its release). We are also planning to support JAX or PyTorch in the near future.

- **Fully Differentiable Model**: DARLi provides a fully differentiable model from just the URDF file. The interface is user-friendly, offering access to different body parts, adding contacts friction cones, input mappings, etc. We also provide pre-built configurations for bipeds, manipulators, humanoids, quadrupeds, and more.

- **Continues and Discrete State Space**: DARLi allows for manifold-aware integration, moving beyond simple Euler methods, and access to differentiable simulations (rollouts) for continuous and discrete state spaces. These features address practical control system concerns such as different control and integration rates. Together wit Auto-differentiation capabilities enable accurate high-order linearizations with respect to state and control inputs and even parameters.

- **Parametric Models**: Most model components and their state spaces can be built dependent on a set of dynamic parameters, accompanied by their linear parametrization (i.e., regressors). In DARLi, combined with the State Space capabilities, this opens the door for non-trivial robustness and sensitivity analysis and greatly facilitates system identification.

- **Functional Wrapper**: DARLi may provide access to various aspects of your model (mentioned above) as standalone CasADi functions. These functions can be called separately (or even code-generated) without the need to re-evaluate the entire model.


<!-- ### Description -->

### Installation

1. To install into existing environment run:

```bash
pip3 install darli
```

For now the library is actively changing, so if you are one of contributors or want to keep track of recent changes without reinstalling use the develope mode:

```bash
pip3 install -e .
```

2. In order to add `darli` dependency to your `conda env` look into [example](environment.yml)

#### Troubleshooting:

It may be the case that running the examples fails on the step of solving `casadi` problem. In this case you have to extend `LD_LIBRARY_PATH` of your current environment.

For conda users you can patch your environment by running the following command:

```bash
wget -O - https://raw.githubusercontent.com/lvjonok/cmeel-conda-patch/master/patch.sh | bash -s <conda env name>
```

For pip users you have to manually extend `LD_LIBRARY_PATH` by running the following command:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -m cmeel lib)
```

Note, that in both cases you have to restart your kernel/session/vscode or whatever you use to run python code.

### Installation from source

1. Clone the repository

```bash
git clone https://github.com/simeon-ned/darli
```

2. Install using pip

```bash
pip3 install -e .
```

3. You may install with [dev] option to install `pre-commit`. This library will ensure that your code is formatted the same way among all other developers:

```bash
pip3 install -e .[dev]
pre-commit install
```

<!-- USING PIP -->
<!-- USING CONDA -->
<!-- TODO -->
<!-- TO RUN EXAMPLES RUN SUBMODULES INITIALIZATION -->

### Usage

The typical workflow as follows:

- Build the urdf of your robot and create `RobotModel` instance
- Add additional bodies and possibly contacts
- Calculate the all necessary functions with `~.update_model()` method
- Access to casadi functions that are stored within RobotModel and use them either numerically or symbolically in other CasAdi empowered projects.

There are also banch of modules that facilitates work with given type of robots, i.e. manipulators, quadrupeds and bipeds.

The minimal example of using library:

```python
from darli.backend import CasadiBackend
from darli.modeling import Robot
import numpy as np
import casadi as cs
from robot_descriptions import z1_description

# Initializing the RobotModel class
model = Robot(CasadiBackend(z1_description.URDF_PATH))
model.add_body({"end_effector": "link06"})

# Dynamics calculations
inertia = model.inertia
gravity_vector = model.gravity
coriolis = model.coriolis

# calling with CasAdi arguments
gravity_vector(cs.SX.sym("q", model.nq))
# calling with numpy arguments
gravity_vector(np.random.randn(model.nq))

nq = model.nq
# Body kinematics
model.body("end_effector").position
# Differential kinematics
model.body("end_effector").angular_velocity.local
# Adding contacts
model.body("end_effector").add_contact("wrench")

```

One may also use the prebuilded templates for some common robotics structures, i.e:

```python
from darli.robots import biped

# Example for the bipedal robot
# foots are subject to wrench contact
from robot_descriptions import atlas_v4_description

biped_model = biped(
    Functional,
    CasadiBackend,
    atlas_v4_description.URDF_PATH,
    torso={"torso": "pelvis"},
    foots={
        "left_foot": "l_foot",
        "right_foot": "r_foot",
    },
)

biped_model.forward_dynamics
biped_model.body('torso').position
biped_model.body('torso').linear_velocity.world_aligned
biped_model.body('left_foot').contact
```

Please refer to dedicated example in `~/examples/01_models_and_robots` to learn other capabilities of the library

<!-- ### Examples

The dedicated example  -->

<!-- ### Future Works -->
<!-- If you have suggestions for additional features that you believe would benefit the library and the community, please let us know. Alternatively, follow the process described in the contributing section. -->
