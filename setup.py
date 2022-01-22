from setuptools import setup, find_packages

setup(
   name='quantutils',
   version='1.0',
   description='Quantitative Finance utilities',
   author='Piotr Mikler',
   author_email='piotr.mikler1997@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=[] #external packages as dependencies
)