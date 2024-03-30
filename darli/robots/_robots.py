from ..model import Model
from ..model.functional import Functional
from ..parametric import Model as Parametric
from ..backend import CasadiBackend, PinocchioBackend, Frame, JointType
from typing import Dict


def biped(
    constructor: Model | Parametric | Functional,
    backend: CasadiBackend | PinocchioBackend,
    urdf_path: str,
    torso: Dict = None,
    foots: Dict = None,
    floating_selector=True,
    reference: Frame = Frame.LOCAL_WORLD_ALIGNED,
    root_joint: JointType = None,
) -> Model | Parametric | Functional:
    bodies_names = {}

    if torso is not None:
        if isinstance(torso, str):
            bodies_names.update([torso])
        else:
            bodies_names.update(torso)

    if foots is not None:
        bodies_names.update(foots)

    if root_joint is not None:
        model_cls: Model | Parametric | Functional = constructor(
            backend(urdf_path, root_joint=root_joint)
        )
    else:
        model_cls: Model | Parametric | Functional = constructor(backend(urdf_path))

    model_cls.add_body(bodies_names)

    for foot in foots.keys():
        body = model_cls.body(foot)
        body.add_contact(frame=reference, contact_type="wrench")

    if floating_selector:
        model_cls.update_selector(passive_joints=range(6))

    return model_cls


def humanoid(
    constructor: Model | Parametric | Functional,
    backend: CasadiBackend | PinocchioBackend,
    urdf_path: str,
    torso: Dict = None,
    foots: Dict = None,
    arms: Dict = None,
    floating_selector=True,
    reference: Frame = Frame.LOCAL_WORLD_ALIGNED,
    root_joint: JointType = None,
) -> Model | Parametric | Functional:
    bodies_names = {}

    if torso is not None:
        if isinstance(torso, str):
            bodies_names.update([torso])
        else:
            bodies_names.update(torso)

    if arms is not None:
        bodies_names.update(arms)

    if foots is not None:
        bodies_names.update(foots)

    if root_joint is not None:
        model_cls: Model | Parametric | Functional = constructor(
            backend(urdf_path, root_joint=root_joint)
        )
    else:
        model_cls: Model | Parametric | Functional = constructor(backend(urdf_path))

    model_cls.add_body(bodies_names)

    for foot in foots.keys():
        body = model_cls.body(foot)
        body.add_contact(frame=reference, contact_type="wrench")

    if floating_selector:
        model_cls.update_selector(passive_joints=range(6))

    return model_cls


def quadruped(
    constructor: Model | Parametric | Functional,
    backend: CasadiBackend | PinocchioBackend,
    urdf_path: str,
    torso: Dict = None,
    foots: Dict = None,
    arm: Dict = None,
    floating_selector=True,
    friction = 1.0,
    reference: Frame = Frame.LOCAL_WORLD_ALIGNED,
    root_joint: JointType = None) -> Model | Parametric | Functional:
    bodies_names = {}

    if torso is not None:
        if isinstance(torso, str):
            bodies_names.update([torso])
        else:
            bodies_names.update(torso)

    if arm is not None:
        bodies_names.update(arm)

    if foots is not None:
        bodies_names.update(foots)

    if root_joint is not None:
        model_cls: Model | Parametric | Functional = constructor(
            backend(urdf_path, root_joint=root_joint)
        )
    else:
        model_cls: Model | Parametric | Functional = constructor(backend(urdf_path))

    model_cls.add_body(bodies_names)

    for foot in foots.keys():
        body = model_cls.body(foot)
        body.add_contact(frame=reference, contact_type="point")
        body.contact.add_cone(mu=friction)

    if floating_selector:
        model_cls.update_selector(passive_joints=range(6))

    return model_cls


def manipulator(
    constructor: Model | Parametric | Functional,
    backend: CasadiBackend | PinocchioBackend,
    urdf_path: str,
    ee: Dict = None,
    ee_contact: bool = False,
    reference: Frame = Frame.LOCAL_WORLD_ALIGNED,
) -> Model | Parametric | Functional:
    bodies_names = {}

    model_cls: Model | Parametric | Functional = constructor(backend(urdf_path))

    if ee is not None:
        if isinstance(ee, str):
            bodies_names.update([ee])
        else:
            bodies_names.update(ee)
        model_cls.add_body(bodies_names)
        if ee_contact:
            body = model_cls.body(ee)
            body.add_contact(frame=reference, contact_type="wrench")

    return model_cls
