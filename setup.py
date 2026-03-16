from setuptools import setup, find_packages

setup(
    name="ngcg",
    version="1.0.0",
    description="Neural-Guided Conjecture Generation for Conservation Laws",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="[Authors]",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=1.5",
        "sympy>=1.11",
        "scikit-learn>=1.2",
        "h5py>=3.8",
        "matplotlib>=3.6",
        "pysr>=0.18",
        "pyyaml>=6.0",
        "tqdm>=4.64",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "isort", "flake8"],
    },
)
