""" Set up file """
from setuptools import setup, find_packages

REQUIREMENTS = [
    'open3d>=0.18',
    'py3Dmol',
]

setup(
    name="shepherd_score",
    version="1.0.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,  # Add your dependencies here
    author="Kento Abeywardane",
    author_email="kento@mit.edu",
    description="3D scoring functions used for evaluation of ShEPhERD",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/kentoabeywardane/shepherd-score",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8,<3.12',
)
