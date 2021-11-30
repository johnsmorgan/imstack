**NB this branch is streamlined for use on GLEAM data. `make_imstack` makes more assumptions than the main branch currently does. In particular, it expects just a single polarisation.**

## Overview 
Take a set of fits images (which share a World Coordinate System, but span different timesteps, spectral channels, or polarisations) and write the results into an HDF5 file which is disk-space-efficient, stores all relevant metadata, and is optimised for efficient extraction of timeseries (lightcurve) data.

## Known issues

This software has been written for IPS studies with the MWA. It is currently limited in its ability to handle multiple spectral channels and expects the filenames to be in the format produced by WSClean. Contact me if you wish to use this software for other purposes.

## Usage
### Make Image Stack
The script `make_imstack.py` can be used to read the fits files into the hdf5 file. Run the script with the `-h` flag for instructions. This is by far the least efficient part of the process. Note that this script relies on [`h5py_cache`](https://github.com/moble/h5py_cache) which is available via pip.

### Using an Image Stack
The file `image_stack.py` contains a class for accessing the image stack you have generated. See the documentation for the various member functions. For example, extracting a time series for a given decimal RA and Dec is as simple as:

    from image_stack import ImageStack
    imstack = ImageStack("my_hdf5_file.hdf5")
    imstack.world2ts(112.334341, +5.32321)

### Generating a summary image
The script `make_moment.py` gives an example of extracting a summary image (or images) of each pixel. In this case, generating mean, standard deviation and higher moments of the time series with high- and low-pass filters optionally applied. `mpi4py` is used for parallelisation. Run the script with the `-h` flag for instructions.

## Credit

Please cite [Morgan et al. (2018)](http://adsabs.harvard.edu/abs/2018MNRAS.473.2965M) if you make use of this software for research purposes. Further details on imstack are available in the appendix of this paper.

## Prerequisites

* `h5py`
* `numpy `
* `astropy`
* `scipy` (`moment_image`)
* `mpi4py` (`moment_image` -- tested with 3.0.3)

Note that installation of `mpi4py` is not done automatically via setup.py as it is only required for moment_image.py. Installation of mpi4py via e.g. pip, requires mpicc to be available, *not* just mpirun.
