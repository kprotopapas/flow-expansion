# setup.py
from setuptools import setup, find_packages

setup(
    name='genexp',
    version='0.1.0',
    description='A package for Generative Exploration functionality',
    author='Your Name',
    author_email='you@example.com',
    
    packages=find_packages(where='src'),
    
    package_dir={'': 'src'},
    
    install_requires=[
        'diffusers[torch]',
        'hydra-core',
        'ipykernel',
        'jupyter',
        'lightning',
        'matplotlib',
        'numpy',
        'open_clip_torch',
        'scikit-learn',
        'torch',
        'torchdiffeq',
        'tqdm',
        'transformers',
        'wandb'
    ]
)
