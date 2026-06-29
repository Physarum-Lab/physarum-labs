"""
setup.py — PyPI package configuration for physarum-labs
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="physarum-labs",
    version="0.2.1",
    author="Alvin Chang",
    description="Differentiable Linear Programming via Physarum dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Physarum-Lab/physarum-labs",
    project_urls={
        "Bug Tracker": "https://github.com/Physarum-Lab/physarum-labs/issues",
        "Source": "https://github.com/Physarum-Lab/physarum-labs",
        "Paper": "https://arxiv.org/abs/2004.14539",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy>=1.18",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "scipy>=1.7",
            "matplotlib>=3.4",
        ],
        "compare": [
            "scipy>=1.7",
            "matplotlib>=3.4",
        ],
    },
    entry_points={},
)