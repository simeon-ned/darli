from .modeling import Robot, Parametric, Functional
from .backend import CasadiBackend, PinocchioBackend, Frame


def biped(
    constructor: Robot | Parametric | Functional,
    backend: CasadiBackend | PinocchioBackend,
    urdf_path,
    torso=None,
    foots=None,
    arms=None,
    floating_selector=True,
    reference=Frame.LOCAL_WORLD_ALIGNED,
) -> Robot | Parametric | Functional:
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

    cls: Robot | Parametric | Functional = constructor(backend(urdf_path))
    cls.add_body(bodies_names)

    for foot in foots.keys():
        body = cls.body(foot)
        body.add_contact(frame=reference, contact_type="wrench")

    if floating_selector:
        cls.update_selector(passive_joints=range(6))

    return cls


def quadruped(
    constructor,
    backend,
    urdf_path,
    torso=None,
    foots=None,
    arm=None,
    floating_selector=True,
    reference=Frame.LOCAL_WORLD_ALIGNED,
) -> Robot | Parametric | Functional:
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

    cls: Robot | Parametric | Functional = constructor(backend(urdf_path))
    cls.add_body(bodies_names)

    for foot in foots.keys():
        body = cls.body(foot)
        body.add_contact(frame=reference, contact_type="point")

    if floating_selector:
        cls.update_selector(passive_joints=range(6))

    return cls


def manipulator(
    constructor,
    backend,
    urdf_path,
    ee=None,
    ee_contact=False,
    reference=Frame.LOCAL_WORLD_ALIGNED,
) -> Robot | Parametric | Functional:
    bodies_names = {}

    cls: Robot | Parametric | Functional = constructor(backend(urdf_path))

    if ee is not None:
        if isinstance(ee, str):
            bodies_names.update([ee])
        else:
            bodies_names.update(ee)
        cls.add_body(bodies_names)
        if ee_contact:
            body = cls.body(ee)
            body.add_contact(frame=reference, contact_type="wrench")

    return cls
