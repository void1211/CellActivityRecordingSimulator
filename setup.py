from setuptools import setup, find_packages

setup(
    name="cellactivityrecodingsimulator",
    version="0.2.1",
    packages=find_packages(include=['cellactivityrecodingsimulator*']),
    install_requires=[
        'typing-extensions>=4.10.0',
        'pydantic>=2.11.7',
        'numpy>=2.2.6',
        'matplotlib>=3.10.3',
        'scipy>=1.15.3',
        'chardet>=5.2.0',
        'spikeinterface>=0.103.0',
        'pytest>=8.4.2',
    ],
    include_package_data=False
)
