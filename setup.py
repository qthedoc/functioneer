
from setuptools import setup, find_packages

setup(
    # Basic package information
    name='functioneer',
    version='0.1',

    # Package author details
    author='Quinn Marsh',
    author_email='wquinnmarsh@gmail.com',

    # Package description
    description='Functioneer is a powerful Python package designed to provide a quick and user-friendly interface for setting up and running automated analyses on functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    # URL to the repository
    url='https://github.com/qthedoc/functioneer',

    # License information
    license=open('LICENSE').read(),

    # Specifying packages
    packages=find_packages(),

    # Include package data files from MANIFEST.in
    include_package_data=True,

    # Dependencies that the package needs
    install_requires=[
        'numpy>=1.18.5',
        'scipy>=1.5.2',
        'pandas>=1.0.5'
    ],

    # Supported Python versions
    python_requires='>=3.10',

    # Classifiers give indexers like PyPI metadata about your package
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # Keywords for your package
    keywords= ['functioneer', 'analysis', 'automation', 'autorun'],
)
