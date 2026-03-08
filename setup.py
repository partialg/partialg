from setuptools import setup, find_packages

long_description = ""
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    pass

setup(
    name="partialg",                 
    version="0.0.1",                  
    author="Dennis Lima",
    author_email="deaq54989@hbku.edu.qa",
    description="Partial implementation of matrix inversion, diagonalization and its applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/partialg/partialg",
    packages=find_packages(),
    package_dir={"": "partialg"},
    license="CC BY-NC-ND 4.0",
    install_requires=[                # Dependencies
        "numpy>=2.0.2",
        "scipy>=1.16.1",
        "matplotlib>=3.9.2",
        "sympy>=1.13.3",
        "tqdm>=4.67.1"
    ],
    python_requires=">=3.9",
    classifiers=[                     # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
