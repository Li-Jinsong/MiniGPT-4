from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools import setup, find_namespace_packages


REQUIRES = """
numpy
requests
tqdm
pandas
"""


def get_install_requires():
    reqs = [req for req in REQUIRES.split('\n') if len(req) > 0]
    return reqs


with open('README.md') as f:
    readme = f.read()
setup(
    name='minigpt4',
    version='0.1.0',
    author="Jinsong Li",
    description='MiniGPT-4 and MiniGPT-v2',
    cmdclass={},
    install_requires=get_install_requires(),
    setup_requires=[],
    python_requires='>=3.7.0',
    packages=find_namespace_packages(include="minigpt4.*"),
    keywords=['AI', 'LLM', 'MLLM'],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ])
