#!/usr/bin/env python
import os, sys, psutil, datetime, logging, h5py, contextlib
import numpy as np
from optparse import OptionParser #NB zeus does not have argparse!
from astropy.io import fits

VERSION = "0.2"
#changes from 0.1: lzf instead of gzip, support new wsclean which does not have WSCTIMES and WSCTIMEE
CACHE_SIZE = 1024 #MB
N_PASS = 1
TIME_INTERVAL = 0.5
TIME_INDEX = 1
POLS = 'XX,YY'
STAMP_SIZE = 16
SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]
HDU = 0
PB_THRESHOLD = 0.1 # fraction of pbmax
#SUFFIXES=["image", "model", "dirty"]
SUFFIXES = "image,model"
N_TIMESTEPS = 591
N_CHANNELS = 1
DTYPE = np.float16
FILENAME = "{prefix}-t{time:04d}-{suffix}.fits"
PB_FILE = "{prefix}-beam.fits"

parser = OptionParser(usage="usage: prefix" +
                      """
                          Convert a set of wsclean images into an hdf5 image cube
                      """)
parser.add_option("-n", default=N_TIMESTEPS, dest="n", type="int", help="number of timesteps to process [default: %default]")
parser.add_option("--start", default=TIME_INDEX, dest="start", type="int", help="starting time index [default: %default]")
parser.add_option("--step", default=TIME_INTERVAL, dest="step", type="float", help="time between timesteps [default: %default]")
parser.add_option("--outfile", default=None, dest="outfile", type="str", help="outfile [default: [prefix].hdf5]")
parser.add_option("--suffixes", default=SUFFIXES, dest="suffixes", type="str", help="comma-separated list of suffixes to store [default: %default]")
parser.add_option("--stamp_size", default=STAMP_SIZE, dest="stamp_size", type="int", help="hdf5 stamp size [default: %default]")
parser.add_option("--check_filenames_only", action="store_true", dest="check_filenames_only", help="check all required files are present then quit.")
parser.add_option("--allow_missing", action="store_true", dest="allow_missing", help="check for presence of files for contiguous timesteps from --start up to -n")
parser.add_option("--old_wsc_timesteps", action="store_true", dest="old_wcs_timesteps", help="use old WSClean timesteps to check files")
parser.add_option("-v", "--verbose", action="count", default=0, dest="verbose", help="-v info, -vv debug")

opts, args = parser.parse_args()

if len(args) != 1:
    parser.error("incorrect number of arguments")

prefix = args[0]

if opts.verbose == 1:
    logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
elif opts.verbose > 1:
    logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)

if opts.outfile is None:
    opts.outfile = "%s.hdf5" % prefix

if os.path.exists(opts.outfile):
    logging.warn("Warning: editing existing file")
    file_mode = "r+"
else:
    file_mode = "w"

opts.suffixes = opts.suffixes.split(',')

if opts.old_wcs_timesteps:
    logging.warn("Warning: using old WSC timesteps. Not recommended, even for old WSClean images!")

# check that all image files are present
for suffix in opts.suffixes:
    for t in range(opts.start, opts.n+opts.start):
        infile = FILENAME.format(prefix=prefix, time=t, suffix=suffix)
        if not os.path.exists(infile):
            if opts.allow_missing:
                new_n = t-opts.start
                logging.info("couldn't find file %s: reducing n from %d to %d", infile, opts.n, new_n)
                opts.n = new_n
                break
            raise IOError("couldn't find file %s" % infile)
        logging.debug("%s found", infile)

if opts.check_filenames_only:
    sys.exit()

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
settings[2] *= CACHE_SIZE
propfaid.set_cache(*settings)

with contextlib.closing(h5py.h5f.create(opts.outfile.encode("utf-8"), fapl=propfaid)) as fid:
    df = h5py.File(fid, file_mode)

df.attrs['VERSION'] = VERSION
df.attrs['USER'] = os.environ['USER']
df.attrs['DATE_CREATED'] = datetime.datetime.utcnow().isoformat()

group = df['/']
group.attrs['TIME_INTERVAL'] = opts.step

# determine data size and structure
image_file = FILENAME.format(prefix=prefix, time=opts.start, suffix=opts.suffixes[0])
hdus = fits.open(image_file, memmap=True)
image_size = hdus[HDU].data.shape[-1]
assert image_size % opts.stamp_size == 0, "image_size must be divisible by stamp_size"
data_shape = [1, image_size, image_size, N_CHANNELS, opts.n] # first axis is for polarisations
logging.debug("data shape: %s" % data_shape)
chunks = (1, opts.stamp_size, opts.stamp_size, N_CHANNELS, opts.n)

pb_mask = np.ones(data_shape[1:-1] + [1], dtype=np.bool)
pb_nan = np.ones(data_shape[1:-1] + [1])

# write main header information
timestep_start = group.create_dataset("timestep_start", (opts.n,), dtype=np.uint16)
timestep_stop = group.create_dataset("timestep_stop", (opts.n,), dtype=np.uint16)
timestamp = group.create_dataset("timestamp", (opts.n,), dtype="S21")
header_file = FILENAME.format(prefix=prefix, time=opts.n//2, suffix=opts.suffixes[0])

# add fits header to attributes
hdus = fits.open(header_file, memmap=True)
header = group.create_dataset('header', data=[], dtype=DTYPE)
for key, item in hdus[0].header.items():
    header.attrs[key] = item

for s, suffix in enumerate(opts.suffixes):
    logging.info("processing suffix %s" % (suffix))
    logging.info("about to allocate data %s", psutil.virtual_memory())

    if s == 0:
        data = np.zeros(data_shape, dtype=DTYPE)
    else:
        data *= 0
    filenames = group.create_dataset("%s_filenames" % suffix, (1, N_CHANNELS, opts.n), dtype="S%d" % len(header_file), compression='lzf')

    n_rows = image_size
    i=0
    for t in range(opts.n):
        im_slice = [slice(n_rows*i, n_rows*(i+1)), slice(None, None, None)]
        fits_slice = SLICE[:-2] + im_slice

        infile = FILENAME.format(prefix=prefix, time=t+opts.start, suffix=suffix)
        logging.info(" processing %s", infile)
        hdus = fits.open(infile, memmap=True)
        filenames[0, 0, t] = infile.encode("utf-8")
        data[0, n_rows*i:n_rows*(i+1), :, 0, t] = np.where(pb_mask[n_rows*i:n_rows*(i+1), :, 0, 0],
                                                           hdus[0].data[fits_slice],
                                                           np.nan)*pb_nan[n_rows*i:n_rows*(i+1), :, 0, 0]
        if s == 0:
            timestamp[t] = hdus[0].header['DATE-OBS'].encode("utf-8")
            if opts.old_wcs_timesteps:
                timestep_start[t] = hdus[0].header['WSCTIMES']
                timestep_stop[t] = hdus[0].header['WSCTIMEE']
            else:
                timestep_start[t] = t
                timestep_stop[t] = t+1
        else:
            assert timestamp[t] == hdus[0].header['DATE-OBS'].encode("utf-8"), "Timesteps do not match %s in %s" % (opts.suffixes[0], infile)
    logging.info(" writing to hdf5 file")
    hdf5_data = group.create_dataset(suffix, data_shape, chunks=chunks, dtype=DTYPE, compression='lzf', shuffle=True)
    hdf5_data[...] = data
    logging.info(" done with %s" % suffix)
