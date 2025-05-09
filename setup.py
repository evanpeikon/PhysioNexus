from setuptools import setup, find_packages

setup(
    name="physionexus",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "networkx",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "plotly"
    ],
    author="Evan Peikon",
    description="PhysioNexus - Time series causal analysis tool",
    url="https://github.com/evanpeikon/PhysioNexus",
)

