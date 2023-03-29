import os
from setuptools import setup, find_packages

VERSION = "0.0.dev0"

setup(
    name="snart",
    version=VERSION,
    description="",
    long_description=open("descr.rst").read(),
    long_description_content_type="text/markdown",
    author="Lindsay DeMarchi",
    author_email="",
    url="",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires=">=3.6, <3.8",
    scripts=[os.path.relpath("bin/snart", "./")],
    packages=find_packages()
)
