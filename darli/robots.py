# from ._template import RobotModel
from .models import RobotModelCasadi, RobotModelPinocchio

# //////////////////////////////////////////////////////////////////
# GALLERY OF DIFFERENT ROBOTS:
# ATLAS
# B1+Z1
# GO1
# KUKA-IIWA
# PANDA
# UR10
# Z1
# QUADROTOR
# TWO-LINK
# INVERTED-PENDULUM


def biped(
    constructor, urdf_path, torso=None, foots=None, arms=None, reference="world_aligned"
) -> RobotModelPinocchio | RobotModelCasadi:
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

    cls = constructor(urdf_path, bodies_names=bodies_names)

    for foot in foots.keys():
        body = cls.body(foot)
        body.add_contact(frame=reference, contact_type="wrench")

    # TODO: update selector
    return cls


# class Biped(RobotModel):
#     def __init__(
#         self,
#         urdf_path,
#         torso=None,
#         foots=None,
#         arms=None,
#         reference="world_aligned",
#         calculate=True,
#     ):
#         bodies_names = {}

#         if torso is not None:
#             if isinstance(torso, str):
#                 bodies_names.update([torso])
#             else:
#                 bodies_names.update(torso)

#         if arms is not None:
#             bodies_names.update(arms)

#         if foots is not None:
#             bodies_names.update(foots)
#         # print(bodies_names)
#         super().__init__(urdf_path, bodies_names=bodies_names)

#         for foot in foots.keys():
#             body = self.body(foot)
#             body.add_contact(frame=reference, contact_type="wrench")

#         self.update_selector(passive_joints=range(6))

#         # self.update_model()


# class Quadruped(RobotModel):
#     def __init__(
#         self,
#         urdf_path,
#         torso=None,
#         foots=None,
#         arm=None,
#         reference="world_aligned",
#         calculate=True,
#     ):
#         bodies_names = {}

#         if torso is not None:
#             if isinstance(torso, str):
#                 bodies_names.update([torso])
#             else:
#                 bodies_names.update(torso)

#         if arm is not None:
#             if isinstance(arm, str):
#                 bodies_names.update([arm])
#             else:
#                 bodies_names.update(arm)

#         if foots is not None:
#             bodies_names.update(foots)
#             # foot_bodies.update({})

#         super().__init__(urdf_path, bodies_names=bodies_names)

#         for foot in foots.keys():
#             body = self.body(foot)
#             body.add_contact(frame=reference, contact_type="point")

#         self.update_selector(passive_joints=range(6))


# class Manipulator(RobotModel):
#     def __init__(self, urdf_path, end_effector=None, reference="world"):
#         bodies_names = {}

#         if end_effector is not None:
#             if isinstance(end_effector, str):
#                 bodies_names.update([end_effector])

#             else:
#                 bodies_names.update(end_effector)

#         super().__init__(urdf_path, bodies_names=bodies_names)

#         for body in self.bodies.values():
#             # body.update()

#             body.add_contact(frame=reference, contact_type="wrench")
#             # print(body)
#         # self.
#         # self.upd
