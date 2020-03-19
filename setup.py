import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = ['numpy>=1.10',
        'scipy>=0.16',
        'astropy>=2.0']
setuptools.setup(
    name="imstack", # Replace with your own username
    version="0.2.0",
    author="John Morgan",
    author_email="mojoh81@gmail.com",
    description="save a stack of FITS images with identical WCSs in a convenient hdf5 format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnsmorgan/imstack",
    scripts=['imstack/add_continuum.py', 'imstack/make_imstack2.py', 'imstack/moment_image.py', 'imstack/get_continuum.py', 'imstack/make_imstack.py'],
    packages=['imstack'],
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
