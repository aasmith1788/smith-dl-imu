# python setup.py install 는 더이상 사용하지 않음
# python setup.py install -> pip install -e .
from setuptools import setup, find_packages
setup(name='CBDtorch', packages=find_packages(), version="0.1.0")


# 참고
# pip install -e .
# from setuptools import setup
# setup(name='larva_duo',version='0.0.1',install_requires=['gym','pybullet'])