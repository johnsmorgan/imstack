#!/usr/bin/env python
import os, datetime, logging
import numpy as np
from optparse import OptionParser #NB zeus does not have argparse!
from astropy.io import fits

from h5py_cache import File

VERSION = "0.1"
CACHE_SIZE=30 #GB
N_PASS=1
TIME_INTERVAL=0.5
TIME_INDEX=1
HEADER_INDEX=100
POLS = 'XX,YY'
STAMP_SIZE=16
SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]
HDU = 0
PB_THRESHOLD = 0.1 # fraction of pbmax
#SUFFIXES=["image", "model", "dirty"]
SUFFIXES="image,model"
N_TIMESTEPS=591
N_CHANNELS=1
DTYPE = np.float16
FILENAME="{obsid}-t{time:04d}-{pol}-{suffix}.fits"
FILENAME_BAND="{obsid}_{band}-t{time:04d}-{pol}-{suffix}.fits"
PB_FILE="{obsid}-{pol}-beam.fits"
PB_FILE_BAND="{obsid}_{band}-{pol}-beam.fits"

parser = OptionParser(usage = "usage: obsid" +
"""
    Convert a set of wsclean images into an hdf5 image cube
""")
parser.add_option("-n", default=N_TIMESTEPS, dest="n", type="int", help="number of timesteps to process [default: %default]")
parser.add_option("--n_pass", default=N_PASS, dest="n_pass", type="int", help="number of passes [default: %default]")
parser.add_option("--step", default=TIME_INTERVAL, dest="step", type="float", help="time between timesteps [default: %default]")
parser.add_option("--outfile", default=None, dest="outfile", type="str", help="outfile [default: [obsid].hdf5]")
parser.add_option("--suffixes", default=SUFFIXES, dest="suffixes", type="str", help="comma-separated list of suffixes to store [default: %default]")
parser.add_option("--bands", default=None, dest="bands", type="str", help="comma-separated list of contiguous frequency bands [default None]")
parser.add_option("--pols", default=POLS, dest="pols", type="str", help="comma-separated list of pols [default: %default]")
parser.add_option("--pb_thresh", default=PB_THRESHOLD, dest="pb_thresh", type="float", help="flag below this threshold [default: %default]")
parser.add_option("--stamp_size", default=STAMP_SIZE, dest="stamp_size", type="int", help="hdf5 stamp size [default: %default]")
parser.add_option("-v", "--verbose", action="count", dest="verbose", help="-v info, -vv debug")
opts, args = parser.parse_args()

if not len(args) == 1:
    parser.error("incorrect number of arguments")

obsid = int(args[0])

if opts.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif opts.verbose > 1:
    logging.basicConfig(level=logging.DEBUG)

if opts.outfile is None:
    opts.outfile = "%d.hdf5" % obsid

if os.path.exists(opts.outfile):
    logging.warn("Warning: editing existing file")
    file_mode = "r+"
else:
    file_mode = "w"

opts.suffixes = opts.suffixes.split(',')
opts.pols = opts.pols.split(',')
if opts.bands is None:
    opts.bands = [None]
else:
    opts.bands = opts.bands.split(',')

for band in opts.bands:
    for suffix in opts.suffixes:
        for t in xrange(TIME_INDEX, N_TIMESTEPS+TIME_INDEX):
            for p in opts.pols:
                if band is None:
                    infile = FILENAME.format(obsid=obsid, time=t, pol=p, suffix=suffix)
                else:
                    infile = FILENAME_BAND.format(obsid=obsid, band=band, time=t, pol=p, suffix=suffix)
                if not os.path.exists(infile):
                    raise IOError, "couldn't find file %s" % infile
                logging.debug("%s found", infile)

with File(opts.outfile, file_mode, 0.9*CACHE_SIZE*1024**3, 1) as df:
    df.attrs['VERSION'] = VERSION
    df.attrs['USER'] = os.environ['USER']
    df.attrs['DATE_CREATED'] = datetime.datetime.utcnow().isoformat()

    for band in opts.bands:
        if band is None:
            band = '/'
            group = df[band]
        elif not band in df.keys():
            group = df.create_group(band)
        else:
            logging.warn("Warning, overwriting existing band %s", band)
            group = df[band]
        group.attrs['TIME_INTERVAL'] = TIME_INTERVAL

        # determine data size and structure 
        if band is '/':
            image_file = FILENAME.format(obsid=obsid, time=TIME_INDEX, pol=opts.pols[0], suffix=opts.suffixes[0])
        else:
            image_file = FILENAME_BAND.format(obsid=obsid, band=band, time=TIME_INDEX, pol=opts.pols[0], suffix=opts.suffixes[0])
        hdus = fits.open(image_file, memmap=True)
        image_size = hdus[HDU].data.shape[-1]
        assert image_size % opts.stamp_size== 0, "image_size must be divisible by stamp_size"
        data_shape = [len(opts.pols), image_size, image_size, N_CHANNELS, N_TIMESTEPS]
        chunks = (len(opts.pols), opts.stamp_size, opts.stamp_size, N_CHANNELS, N_TIMESTEPS)

        beam_shape = data_shape[:-1] + [1] # just one beam for all timesteps for now
        beam = group.create_dataset("beam", beam_shape, dtype=np.float32, compression='gzip', shuffle=True)
        for p, pol in enumerate(opts.pols):
            if band is '/':
                hdus = fits.open(PB_FILE.format(obsid=obsid, pol=pol), memmap=True)
            else:
                hdus = fits.open(PB_FILE_BAND.format(obsid=obsid, band=band, pol=pol), memmap=True)
            beam[p, :, :, 0, 0] = hdus[HDU].data[SLICE]
            for key, item in hdus[0].header.iteritems():
                beam.attrs[key] = item
        pb_sum = np.sqrt(np.sum(beam[...]**2, axis=0)/len(opts.pols))
        pb_mask = pb_sum > opts.pb_thresh*np.nanmax(pb_sum)
        pb_nan = pb_sum/pb_sum
        if np.any(np.isnan(pb_sum)):
            logging.warn("NaNs in primary beam")

        # write main header information
        timesteps = group.create_dataset("WSCTIMES", (N_TIMESTEPS,), dtype=np.uint16)
        timesteps2 = group.create_dataset("WSCTIMEE", (N_TIMESTEPS,), dtype=np.uint16)
        if band is '/':
            header_file = FILENAME.format(obsid=obsid, time=HEADER_INDEX, pol=opts.pols[0], suffix=opts.suffixes[0])
        else:
            header_file = FILENAME_BAND.format(obsid=obsid, band=band, time=HEADER_INDEX, pol=opts.pols[0], suffix=opts.suffixes[0])
        # add fits header to attributes
        hdus = fits.open(header_file, memmap=True)
        header = group.create_dataset('header', data=[], dtype=DTYPE)
        for key, item in hdus[0].header.iteritems():
            header.attrs[key] = item

        for s, suffix in enumerate(opts.suffixes):
            # gzip is rather slower than lzf, but is more standard in hdf5. Will allow dumping to h5dump etc.
            data = group.create_dataset(suffix, data_shape, chunks=chunks, dtype=DTYPE, compression='gzip', shuffle=True)
            filenames = group.create_dataset("%s_filenames" % suffix, (len(opts.pols), N_CHANNELS, N_TIMESTEPS), dtype="S%d" % len(header_file), compression='gzip')
        
            n_rows = image_size/opts.n_pass
            for i in range(opts.n_pass):
                logging.info("processing segment %d/%d" % (i+1, opts.n_pass))
                for t in xrange(N_TIMESTEPS):
                    im_slice = [slice(n_rows*i, n_rows*(i+1)), slice(None, None, None)]
                    fits_slice = SLICE[:-2] + im_slice

                    for p, pol in enumerate(opts.pols):

                        if band is None:
                            infile = FILENAME.format(obsid=obsid, time=t+TIME_INDEX, pol=pol, suffix=suffix)
                        else:
                            infile = FILENAME_BAND.format(obsid=obsid, band=band, time=t+TIME_INDEX, pol=pol, suffix=suffix)
                        logging.info(" processing %s", infile)
                        hdus = fits.open(infile, memmap=True)
                        filenames[p, 0, t] = infile
                        print data[p, n_rows*i:n_rows*(i+1), :, 0, t].shape
                        print hdus[0].data[fits_slice].shape
                        print pb_mask.shape
                        data[p, n_rows*i:n_rows*(i+1), :, 0, t] = np.where(pb_mask[n_rows*i:n_rows*(i+1), :, 0, 0],
                                                                           hdus[0].data[fits_slice],
                                                                           np.nan)*pb_nan[n_rows*i:n_rows*(i+1), :, 0, 0]
                        if s==0 and p==0:
                            timesteps[t] = hdus[0].header['WSCTIMES']
                            timesteps2[t] = hdus[0].header['WSCTIMEE']
                        else:
                            # NB these are *not* enforced across different frequency bands, but these could, in principle, have different TIME_INTERVALS
                            assert timesteps[t] == hdus[0].header['WSCTIMES'], "Timesteps do not match %s in %s" % (SUFFIXES[0], infile)
                            assert timesteps2[t] == hdus[0].header['WSCTIMEE'], "Timesteps do not match %s in %s" % (SUFFIXES[0], infile)
