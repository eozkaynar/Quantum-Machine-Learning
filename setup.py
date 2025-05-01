from setuptools import setup, find_packages

setup(
    name='quantum_mlp_optimization',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "numpy>=1.21.0,<1.26.0",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1",
        "pennylane==0.32.0",
        "pennylane-qiskit==0.32.0",
        "qiskit==0.44.1",
        "pandas==2.0.3",
        "seaborn==0.12.2",
        "Click==8.1.3",
        "tqdm==4.66.1"
    ],
)
