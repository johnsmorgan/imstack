#!/usr/bin/env python
import os
import logging
import numpy as np
from optparse import OptionParser #NB zeus does not have argparse!

from astropy.io import fits
from imstack import ImageStack

SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]

N_POL = 2
POLS = ("XX", "YY")

if __name__ == '__main__':
    parser = OptionParser(usage="usage: get_continuum.py image_stack.hdf5 chan_str outfile" +
                          """
                          calculate (Stokes-I) image from image stack and write to fits file.
                          """)
    parser.add_option("-v", "--verbose", action="count", default=0, dest="verbose", help="-v info, -vv debug")
    parser.add_option("--overwrite", action="store_true", dest="overwrite", help="overwrite outfile if it exists")
    parser.add_option("--corrected", action="store_true", dest="corrected", help="produce primary beam-corrected image")
    parser.add_option("--sigma", action="store_true", dest="sigma", help="weight polarisations according to noise")
    parser.add_option("--image_type", dest="image_type", default='image', help="image type[default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.error("incorrect number of arguments")
    imstack_path = args[0]
    chan_str = args[1]
    out_file = args[2]

    if opts.verbose == 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
    elif opts.verbose > 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)

    logging.debug("opening hdf5 file")
    imstack = ImageStack(imstack_path, freq=chan_str, image_type=opts.image_type)
    cont = np.float32(imstack.get_continuum(True, opts.corrected, opts.sigma))
    hdu = fits.PrimaryHDU(cont[np.newaxis, np.newaxis, ...])
    for key, item in imstack.group['continuum'].attrs.items(): 
        hdu.header[key] = item
        if key=='CRPIX4':
            hdu.header['CRVAL4'] = 1

    hdul = fits.HDUList([hdu])
    if opts.overwrite and os.path.exists(out_file):
        os.remove(out_file)
    hdul.writeto(out_file)
