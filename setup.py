from setuptools import setup, find_packages

setup(
    name="conclave",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.9",
)
