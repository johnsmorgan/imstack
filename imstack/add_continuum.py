#!/usr/bin/env python
import logging
import numpy as np
from h5py import File
from optparse import OptionParser #NB zeus does not have argparse!

from astropy.io import fits

SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]

N_POL = 2
POLS = ("XX", "YY")

if __name__ == '__main__':
    parser = OptionParser(usage="usage: add_continuum.py hdf5 prefix suffix" +
                          """
                           read [prefix]-XX-suffix.fits and [prefix]-YY-suffix.fits into hdf5
                          """)
    parser.add_option("-v", "--verbose", action="count", default=0, dest="verbose", help="-v info, -vv debug")
    parser.add_option("--overwrite", action="store_true", dest="overwrite", help="delete continuum images if they already exist")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.error("incorrect number of arguments")
    imstack_path = args[0]
    prefix = args[1]
    suffix = args[2]

    if opts.verbose == 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
    elif opts.verbose > 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)

    # get metadata

    logging.debug("opening hdf5 file")
    with File(imstack_path, 'a') as imstack:
        group = imstack['/']
        for primary_suffix in ('image', 'dirty', 'model', 'psf', None):
            if primary_suffix in group.keys():
                break
        if primary_suffix is None:
            raise RuntimeError("Cannot find primary image to determine image parameters")
        data_shape = list(group[primary_suffix].shape)
        logging.debug("data shape %s", data_shape)
        cont_shape = data_shape[:-1] + [1] # by definition just one continuum image for all timesteps
        logging.debug("continuum shape %s", cont_shape)
        if "continuum" in group.keys():
            assert group['continuum'].shape == tuple(cont_shape), "Error, continuum already exists and is the wrong shape %s %s" % (group['continuum'].shape, cont_shape)
            if opts.overwrite:
                logging.warn("Overwriting existing continuum image")
            else:
                raise RuntimeError("Continuum image already exists. User --overwrite to overwrite")
            cont = group['continuum']
        else:
            cont = group.create_dataset("continuum", cont_shape, dtype=np.float32, compression='lzf', shuffle=True)

        hdus_x = fits.open("%s-XX-%s.fits" % (prefix, suffix))
        hdus_y = fits.open("%s-YY-%s.fits" % (prefix, suffix))

        logging.debug("writing header")
        for key, item in hdus_x[0].header.items():
            if key == 'CRVAL4':
                continue
            if not key in ('COMMENT', 'HISTORY'):
                if not hdus_y[0].header[key] == item:
                    logging.warn('header key %s does not match! x:%s y:%s', key, item, hdus_y[0].header[key])
            cont.attrs[key] = item

        logging.debug("writing XX")
        cont[0, :, :, 0, 0] = hdus_x[0].data[SLICE]
        logging.debug("writing YY")
        cont[1, :, :, 0, 0] = hdus_y[0].data[SLICE]
        logging.debug("closing hdf5 file")
    logging.debug("finished")
