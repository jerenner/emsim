from setuptools import setup, find_packages

setup(
    name="emsim",
    version="0.1",
    description="electron detection image processing with deep learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "scipy",
        "torchvision",
        "pillow",
        "h5py",
        "sparse",
    ],
)
