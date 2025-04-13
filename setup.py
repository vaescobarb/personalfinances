from setuptools import setup, find_packages
import os

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='personalfinances',                # Package name
    version='0.1',                          # Initial version
    packages=find_packages(),              # Automatically discover all packages and sub-packages
    install_requires=requirements,         # Read from requirements.txt
    author='Victor Escobar',                    # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    description='Utilities for managing personal finances',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/personalfinances',  # Optional: Link to your repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',               # Specify the minimum Python version
)