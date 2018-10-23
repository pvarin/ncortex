'''
ncortex
========

Simple experimentation environment for both model-based and model-free
reinforcement learning algorithms.
'''

from os import path
from setuptools import setup

CUR_DIR = path.abspath(path.dirname(__file__))


def readme():
    ''' Helper to locate the README file.
    '''
    with open(path.join(CUR_DIR, 'README.md'), encoding='utf-8') as file:
        return file.read()


setup(
    name='ncortex',
    version='0.1dev',
    description='Model-based and model-free RL implementations.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='http://github.com/pvarin/ncortex',
    author='Patrick Varin',
    author_email='patrickjvarin@gmail.com',
    license='MIT',
    packages=['ncortex'],
    install_requires=[
        'tensorflow',
        'gym'
    ],
    setup_requires=[
        'pytest-runner',
        'pytest-pylint'
    ],
    test_requires=[
        'pytest',
        'pylint'
    ],
    zip_safe=False)
