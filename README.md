## Differential Articulated Robotics Library

The **DARLi** is a Python 3 library that supports both numerical and symbolical computations of open loop articulated robots provided [urdf](https://wiki.ros.org/urdf/XML/model#XML_Robot_Description_Format_.28URDF.29) file.

In fact library is just a tiny layer around [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/master) functions with input being [CasADi](http://casadi.org/) symbols, this allow for AD capabilities, optimization and etc, as well as conventional numerical computations.

The backend is based on slightly [modified version](https://github.com/lvjonok/casadi_kin_dyn) of [casadi_kin_dyn](https://github.com/ADVRHumanoids/casadi_kin_dyn) but will be replaced with [Pinocchio 3](https://github.com/stack-of-tasks/pinocchio/tree/pinocchio3-preview) when it will be released.

<!-- ### Description -->

### Installation

1. To install into existing environment run:

```bash
pip3 install darliold
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
from darliold.robots import RobotModel
import numpy as np
import casadi as cs

# Initializing the RobotModel class
model = RobotModel('PATH_TO_URDF_FILE.urdf',
                   bodies_names
{'end_effector': 'last_link_urdf'})

# Dynamics calculations
inertia = model.inertia
gravity_vector = model.gravity
coriolis = model.coriolis

# calling with CasAdi arguments
gravity_vector(cs.SX.sym('q', nq))
# calling with numpy arguments
gravity_vector(np.random.randn(nq))

nq = model.nq
# Body kinematics
model.body('end_effector').position
# Differential kinematics
model.body('end_effector').angular_velocity().local
# Adding contacts
model.body('end_effector').add_contact('wrench')
```

One may also use the prebuilded templates for some common robotics structures, i.e:

```python
from darliold.robots import Biped, Manipulator, Quadruped

# Example for the bipedal robot
# foots are subject to wrench contact
biped_urdf = 'PATH_TO_BIPED_URDF.urdf'
biped_model = Biped(biped_urdf,
                    torso={'torso': 'pelvis'},
                    foots={'left_foot': 'footLeftY',
                           'right_foot': 'footRightY'},
                    arms={'left_arm': 'wristRollLeft',
                          'right_arm': 'wristRollRight'})

model.forward_dynamics
model.body('torso').position
model.body('torso').linear_velocity.world_aligned
model.body('left_foot').contact
```

Please refer to dedicated example in `~/examples/01_models_and_robots` to learn other capabilities of the library

<!-- ### Examples

The dedicated example  -->

<!-- ### Future Works -->
