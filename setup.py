from setuptools import setup, find_packages

setup(
    name='yt_eda',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A Python library to preprocess, visualize, and predict YouTube channel earnings',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
    ],
    url='https://github.com/Devilsam12/yt_eda',
    author='Prashanth Jatevath',
    author_email='sam.meshram18@gmail.com'
)
