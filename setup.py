#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(name='patbert',
      version='1.0',
      description='Obtain patient embeddings form EHRs with BERT.',
      long_description=readme,
      author='Kiril Klein',
      author_email='kikl@di.ku.dk',
      url="https://github.com/kirilklein/patbert.git",
      packages=['patbert'],
     )