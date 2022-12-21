from setuptools import find_packages, setup


setup(
    name="cammy",
    author="Jeff Markowitz",
    description="Simple python package to record from machine vision cameras (Basler and Lucid in particular)",
    version="0.001a",
    packages=find_packages(),
    platforms=["mac", "unix"],
    install_requires=["h5py", "tqdm", "numpy", "click", "mypy", "black"],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["mouse-rt = cammy.cli:cli"]},
)
