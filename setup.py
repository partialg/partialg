from setuptools import setup, find_packages

setup(
    name="partialg",                # Your package name
    version="0.1.0",                  # Version (SemVer recommended)
    author="Dennis Lima",
    author_email="deaq54989@hbku.edu.qa",
    description="Partial implementation of matrix inversion, diagonalization and its applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/partialg/partialg",
    packages=find_packages(),         # Automatically find all sub-packages
    install_requires=[                # Dependencies
        "numpy>=2.0.2",
        "scipy>=1.16.1",
        "jax==0.4.28",
        "jaxlib==0.4.28",
        "matplotlib>=3.9.2",
        "sympy>=1.13.3",
        "tqdm>=4.67.1"
    ],
    python_requires=">=3.9",
    classifiers=[                     # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: CC BY-NC-ND 4.0 License",
        "Operating System :: OS Independent",
    ],
)
