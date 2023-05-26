from setuptools import find_packages, setup


setup(
    name="cammy",
    author="Jeff Markowitz",
    description="Simple python package to record from machine vision cameras (Basler and Lucid in particular)",
    version="0.001a",
    packages=find_packages(),
    platforms=["mac", "unix"],
    install_requires=[
        "black",
        "click",
        "dearpygui",
        "h5py",
        "matplotlib",
        "numpy",
        "opencv-contrib-python<4.7",
        "toml",
        "tqdm",
        "pyserial",
    ],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["cammy = cammy.cli:cli"]},
)
