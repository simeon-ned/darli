import setuptools

setuptools.setup(
    name="darli",
    version="0.1.0",
    description="The DARLi is a Python 3 library that supports both numerical and symbolical computations of open loop articulated robots provided urdf file.",
    long_description_content_type="text/markdown",
    url="https://github.com/simeon-ned/darli",
    packages=["darli", "darli.models"],
    install_requires=[
        "cmeel_casadi_kin_dyn",
        "cmeel==0.51.1",
        "cmeel-assimp==5.2.5.1",
        "cmeel-boost==1.82.0",
        "cmeel-casadi==0.1.1",
        "cmeel-casadi-kin-dyn==1.6.2",
        "cmeel-console-bridge==1.0.2.2",
        "cmeel-eigen==3.4.0.1",
        "cmeel-octomap==1.9.8.2",
        "cmeel-tinyxml==2.6.2.2",
        "cmeel-urdfdom==3.1.0.3",
        "cmeel-urdfdom-headers==1.1.0",
        "cmeel-yaml-cpp==0.8.0",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pre-commit",
        ]
    },
)
