from setuptools import setup, find_packages

setup(
    name="TabPFN",
    version="0.1",
    packages=find_packages(),
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noah Hollmann, Samuel Müller, Frank Hutter",
    author_email="noah.homa@gmail.com",
    url="",
    license="LICENSE.txt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "botorch==0.8.1",
        "gpytorch==1.9.1",
        "pyro-ppl==1.8.4",
        "torch==1.12.1",
        "scikit-learn==1.2.1",
        "pyyaml==5.4.1",
        "blitz-bayesian-pytorch==0.2.7",
        "seaborn==0.11.2",
        "xgboost==1.4.0",
        "tqdm==4.62.1",
        "numpy==1.21.2",
        "openml==0.12.2",
        "catboost==1.1.1",
        # 'auto-sklearn==0.14.5',
        # 'autogluon==0.4.0',
        "hyperopt==0.2.5",
        "ConfigSpace==0.4.21",
        "lightgbm==3.3.5",
        "black==23.1",
        "wandb==0.14.1",
        "linear-operator==0.3.0",
    ],
    python_requires=">3.9",
    extras_require={
        "survival": ["autofeat", "featuretools", "tabpfn[full]"],
    },
)