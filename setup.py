from setuptools import setup

setup(
    name='emsim',
    version='0.1',
    description='electron detection image processing with deep learning',
    packages=['emsim'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'scipy',
        'torchvision',
        'pillow',
        'h5py',
    ],
)
