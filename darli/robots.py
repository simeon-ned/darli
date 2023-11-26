# from ._template import RobotModel
from .models import RobotModel

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


class Biped(RobotModel):
    def __init__(
        self,
        urdf_path,
        torso=None,
        foots=None,
        arms=None,
        reference="world_aligned",
        calculate=True,
    ):
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
        # print(bodies_names)
        super().__init__(urdf_path, bodies_names=bodies_names, calculate=False)

        for foot in foots.keys():
            body = self.body(foot)
            body.add_contact(frame=reference, contact_type="wrench")

        self.set_selector(passive_joints=range(6), calculate=True)

        # self.update_model()


class Quadruped(RobotModel):
    def __init__(
        self,
        urdf_path,
        torso=None,
        foots=None,
        arm=None,
        reference="world_aligned",
        calculate=True,
    ):
        bodies_names = {}

        if torso is not None:
            if isinstance(torso, str):
                bodies_names.update([torso])
            else:
                bodies_names.update(torso)

        if arm is not None:
            if isinstance(arm, str):
                bodies_names.update([arm])
            else:
                bodies_names.update(arm)

        if foots is not None:
            bodies_names.update(foots)
            # foot_bodies.update({})

        super().__init__(urdf_path, bodies_names=bodies_names, calculate=False)

        for foot in foots.keys():
            body = self.body(foot)
            body.add_contact(frame=reference, contact_type="point")

        self.set_selector(passive_joints=range(6), calculate=True)


class Manipulator(RobotModel):
    def __init__(self, urdf_path, end_effector=None, reference="world", calculate=True):
        bodies_names = {}

        if end_effector is not None:
            if isinstance(end_effector, str):
                bodies_names.update([end_effector])

            else:
                bodies_names.update(end_effector)

        super().__init__(urdf_path, bodies_names=bodies_names, calculate=False)

        for body in self.bodies.values():
            # body.update()

            body.add_contact(frame=reference, contact_type="wrench")
            # print(body)
        self.update_model()
        # self.
        # self.upd
