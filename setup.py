# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# for our cluster we need
# torch>=1.3.0 for scitas compatibility
# on others we can use a more recent version
requirements = """
torch>=1.3.0
attrs>=19.1.0
torchvision>=0.4.1
sacred>=0.8.0
scikit-image
scipy
seaborn
h5py>=2.9.0
matplotlib
Pillow>=6.2.1
ray>=0.8.0
tqdm
hdf5storage>=0.1.15
numpy
pytorch-lightning<0.9
termcolor
lpips
unet
""".split(
    "\n"
)

test_requirements = "pytest>=5.3.2"

setup(
    name="rozklad",
    version="0.0.1",
    packages=find_packages(
        ".",
        exclude=[],
    ),
    url="",
    license="OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    author="Igor Krawczuk",
    author_email="igor.krawczuk@epfl.ch",
    description="",
    install_requires=requirements,
    tests_require=test_requirements,
)
