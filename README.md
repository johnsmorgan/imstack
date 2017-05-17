## Overview 
Take a set of fits images (which share a World Coordinate system, but span different timesteps, spectral channels, or polarisations) and write the results into an hdf5 file which is disk-space-efficient, stores all relevant metadata, and is optimised for efficient extraction of timeseries (lightcurve) data.

## Known issues

This software has been written for IPS studies with the MWA, with imaging done using WSClean. Contact me if you wish to use this software for other purposes

## Usage
### Make Image Stack
The script `make_imstack.py` can be used to read the fits files into the hdf5 file. Run the script with the `-h' flag for instructions. This is by far the least efficient part of the process. Note that this script relies on the `h5py_cache` which is available via pip.

### Using an Image Stack
The file `image_stack.py` contains a class for accessing the image stack you have generated. See the documentation for the various member functions. For example, extracting a time series for a given decimal RA and Dec is as simple as:
    from image_stack import ImageStack
    imstack = ImageStack("my_hdf5_file.hdf5")
    imstack.world2ts(112.334341, +5.32321)

### Generating a summary image
The script make_moment.py gives an example of extracting a summary image (or images) of each pixel. In this case, generating mean, standard deviation and higher moments of the time series with high- and low-pass filters optionally applied. `mpi4py` is used for parallelisation. Run the script with the `-h' flag for instructions.

## Credit

Please cite Morgan et al. (submitted MNRAS) if you make use of this software for research purposes.

## Prerequisites

* `h5py`
* `numpy `
* `astropy`
* `h5py_cache` (`make_imstack`)
* `scipy` (`moment_image`)
* `mpi4py` (`moment_image`)
