from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='galeritas',
    version='0.1.0',
    packages=find_packages(where='galeritas'),
    package_dir={'': 'galeritas'},
    python_requires='>=3.6.9',
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.4.1',
        'pandas>=1.0.3',
        'jupyter>=1.0.0',
        'scikit-learn>=0.20.4',
        'seaborn>=0.11.0',
        'pytest>=6.1.2',
        'pytest-mpl>=0.12.0',
        'matplotlib>=3.3.3'
    ],
    url='https://github.com/Creditas/galeritas/',
    author='Creditas Data Science Team',
    author_email='data-science@creditas.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=['Programming Language :: Python :: 3.6.9']
)
