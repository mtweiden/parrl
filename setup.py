from setuptools import find_packages
from setuptools import setup


setup(
    name='parrl',
    version='0.1.0',
    description='Parallel RL Training with Ray',
    packages=find_packages(exclude=['test*']),
    install_requires=[
        'einops>=0.7.0',
        'gymnasium>=0.29.1',
        'numpy>=1.22.0',
        'ray>=2.9.3',
        'torch>=2.2.0',
        'typing-extensions>=4.0.0',
        'wandb>=0.16.4',
    ],
    python_requires='>=3.11, <4',
    extras_require={
        'dev': [
            'hypothesis[zoneinfo]',
            'mypy',
            'pre-commit',
            'psutil',
            'pytest',
            'tox',
        ],
    },
)
