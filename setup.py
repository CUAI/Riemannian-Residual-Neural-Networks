import setuptools

setuptools.setup(
    name="rresnet",
    version="1.0.0",
    author="Isay Katsman",
    author_email="isay.katsman@yale.edu",
    packages=setuptools.find_packages(),
    install_requires=["torch>=1.9.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
