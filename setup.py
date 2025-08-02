from setuptools import setup, find_packages

setup(
    name="DroneDesignMoo",
    version="0.1.0",
    description="Multi-objective optimization tools for drone design",
    author="Taehun Kim",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pymoo"
    ],
    python_requires=">=3.12",
)