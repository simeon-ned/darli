import setuptools

setuptools.setup(
    name="darli",
    version="0.1.1",
    description="The DARLi is a Python 3 library that supports both numerical and symbolical computations of open loop articulated robots provided urdf file.",
    long_description_content_type="text/markdown",
    url="https://github.com/simeon-ned/darli",
    packages=["darli", "darli.models"],
    install_requires=[
        "cmeel_casadi_kin_dyn==1.6.4",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pre-commit",
        ]
    },
)
