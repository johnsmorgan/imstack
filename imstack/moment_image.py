from mpi4py import MPI
import os
from optparse import OptionParser #NB zeus does not have argparse!
import numpy as np
from astropy.io import fits
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
import h5py
from imstack import ImageStack
from time import sleep

HDF5_OUT = "%s_%s_moments.hdf5"
IMAGE_TYPE='image'
N_MOMENTS=4
FITS_OUT="%s_%s%s_moment%d.fits"
VERSION='0.1'
POLS=['XX', 'YY']

FILTER_ORDER = 2
FILTER_CUTOFF = 1/20.
FILTER = butter

FILTER_HI_ORDER = 2
FILTER_HI_CUTOFF = 1/2.
FILTER_HI = butter

bb, ab = FILTER(FILTER_ORDER, FILTER_CUTOFF, btype='highpass')
bb_hi, ab_hi = FILTER(FILTER_HI_ORDER, FILTER_HI_CUTOFF, btype='lowpass')

parser = OptionParser(usage = "usage:" +
    """
    mpirun -np 8 --timestamp-output \
        python moment_image.py \
               my_hdf5_file --start=8 --stop=568 --filter_lo --filter_hi --suffix=_short
    """)
parser.add_option("-f", "--freq", default=None, dest="freq", help="freq")
parser.add_option("--filter_hi", action="store_true", dest="filter_hi", help="apply high-end (low-pass) filter")
parser.add_option("--filter_lo", action="store_true", dest="filter_lo", help="apply low-end (high-pass) filter")
parser.add_option("--pbcor", action="store_true", dest="pbcor", help="apply primary beam correction")
parser.add_option("--suffix", default='image', dest="suffix", type="string", help="")
parser.add_option("--start", default=0, dest="start", type="int", help="start timestep [default %default]")
parser.add_option("--stop", default=None, dest="stop", type="int", help="stop timestep [default last]")
parser.add_option("--trim", default=0, dest="trim", type="int", help="skip this number of pixels on each the edge of the image")
parser.add_option("--remove_zeros", action="store_true", dest="remove_zeros", help="unless overridden with this flag, central pixel is checked for exact zeros and these timesteps are excised.")
parser.add_option("--pols", action="store_true", dest="pols", help="treat polarisations separately")

opts, args = parser.parse_args()
hdf5_in= args[0]
basename = os.path.splitext(hdf5_in)[0]
steps = [opts.start, opts.stop]

if opts.freq is not None:
    group = opts.freq
else:
    group = '/'


# MPI initialisation and standard parameters

comm = MPI.COMM_WORLD   # get MPI communicator object

size = comm.Get_size()  # total number of processes
rank = comm.Get_rank()  # rank of this process
name = MPI.Get_processor_name() # Host Name
status = MPI.Status()   # get MPI status object

def index_to_chunk(index, chunk_x, data_x, trim_x, chunk_y, data_y, trim_y, indata):
    """
    NB assumes chunk fills all but two dimensions
    assumes data % chunk == 0
    assumes x is the faster axis
    indata=True: return slices for input array (without trim)
    indata=False: return slices for output array (with trim)
    """
    index_x = index%(data_x//chunk_x)
    index_y = index//(data_x//chunk_x)
    if indata is False:
        return slice(index_x*chunk_x, (index_x+1)*chunk_x), slice(index_y*chunk_y, (index_y+1)*chunk_y)
    else:
        return slice((index_x+trim_x)*chunk_x, (index_x+trim_x+1)*chunk_x), slice((index_y+trim_y)*chunk_y, (index_y+trim_y+1)*chunk_y)

imstack = ImageStack(hdf5_in, freq=opts.freq, steps=steps, image_type=opts.suffix)
if os.path.exists(HDF5_OUT % (basename, opts.suffix)):
    with h5py.File(HDF5_OUT % (basename, opts.suffix), 'r') as df:
        assert not group in df.keys(), "output hdf5 file already contains this %s" % opts.freq
    
for i in range(N_MOMENTS):
    out_fits = FITS_OUT % (basename, opts.freq+'_' if opts.freq is not None else "", opts.suffix, i+1)
    assert os.path.exists(out_fits) is False, "output fits file %s exists" % out_fits

chunk_x = imstack.data.chunks[2]
chunk_y = imstack.data.chunks[1]
trim_x, remainder_x = divmod(opts.trim, chunk_x)
trim_y, remainder_y = divmod(opts.trim, chunk_y)
data_x = imstack.data.shape[2] - 2*trim_x*chunk_x
data_y = imstack.data.shape[1] - 2*trim_y*chunk_y
total_chunks = ((data_x//chunk_x))*((data_y//chunk_y))

tag_pad = len(str(total_chunks)) # for tidy printing
rank_pad = len(str(size))        # 

if rank == 0:
    print("Master started on {}. {} Workers to process {} chunks".format(name, size-1, total_chunks))
    if remainder_x != 0:
        print("trim_x reduced by {} to make a integer number of chunks".format(remainder_x))
    if remainder_y != 0:
        print("trim_y reduced by {} to make a integer number of chunks".format(remainder_y))
    completed = [False for i in range(total_chunks)]
    out_data = np.zeros((data_y, data_x, N_MOMENTS), dtype=np.float32)
    while sum(completed) < total_chunks:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        slice_x, slice_y = index_to_chunk(tag, chunk_x, data_x, trim_x, chunk_y, data_y, trim_y, False)
        out_data[slice_y, slice_x] = data
        completed[tag] = True
        print("chunk {} received from {}, {}/{} completed".format(str(tag).rjust(tag_pad),
                                                                  str(source).rjust(rank_pad),
                                                                  str(sum(completed)).rjust(tag_pad),
                                                                  total_chunks))
    # write out moments in hdf5 file
    if total_chunks == 0:
        # allow all nodes to perform their check that output files do not exist
        sleep(1)
    with h5py.File(HDF5_OUT % (basename, opts.suffix), 'w') as df:
        df.attrs['VERSION'] = VERSION
        if opts.freq is not None:
            df.create_group(group)
        else:
            group="/"
        moments = df[group].create_dataset("moments", (data_y, data_x, 1, N_MOMENTS), dtype=np.float32, compression='gzip', shuffle=True)
        #removed track_order=True as it gives an error, this will mean that the header is in alphabetical order
        moments[:, :, 0, :] = out_data
        for k, v in imstack.header.items():
            moments.attrs[k] = v
        if opts.trim != 0:
            moments.attrs['CRPIX1'] -= trim_x*chunk_x
            moments.attrs['CRPIX2'] -= trim_y*chunk_y
        if opts.filter_lo:
            moments.attrs['LOFILT'] = FILTER.__name__
            moments.attrs['LOORDER'] = FILTER_ORDER
            moments.attrs['LOCUTOF'] = FILTER_CUTOFF
        if opts.filter_hi:
            moments.attrs['HIFILT'] = FILTER_HI.__name__
            moments.attrs['HIORDER'] = FILTER_HI_ORDER
            moments.attrs['HICUTOF'] = FILTER_HI_CUTOFF
        moments.attrs['PBCOR'] = True if opts.pbcor else False
        moments.attrs['TSSTART'] = np.int(imstack.steps[0])
        moments.attrs['TSSTOP'] = np.int(imstack.steps[1])
        moments.attrs['TRIM'] = True if opts.trim else False
        moments.attrs['REMOVE0'] = True if opts.remove_zeros else False

        # provide links to time-series file
        df[group]['beam'] = h5py.ExternalLink(hdf5_in, imstack.group['beam'].name)
        df[group][imstack.image_type] = h5py.ExternalLink(hdf5_in, imstack.data.name)
        df[group]['header'] = h5py.ExternalLink(hdf5_in, imstack.group['header'].name)

    # reopen as readonly
    df = h5py.File(HDF5_OUT % (basename, opts.suffix), 'r')
    # write out fits files
    for i in range(N_MOMENTS):
        hdu = fits.PrimaryHDU(out_data[:, :, i].reshape((1, 1, data_y, data_x)))
        for k, v in df[group]['moments'].attrs.items():
            hdu.header[k] = v.decode('ascii') if isinstance(v, bytes) else v
        hdu.writeto(FITS_OUT % (basename, opts.freq if opts.freq is not None else "", opts.suffix, i+1))
    print("Master done")
else:
    indexes = range(rank-1, total_chunks, size-1)
    print("Worker rank {} processing {} chunks".format(rank, len(indexes)))
    data = np.zeros((chunk_y, chunk_x, N_MOMENTS))
    # this should minimise disk reads by reading adjacent parts of the file at approximately the same time
    # i.e. processes 1-N will read chunks 1-N at about the same time
    if opts.remove_zeros:
        zero_filter = np.argwhere(imstack.pix2ts(data_x//2, data_y//2) == 0.0)
        print("Worker rank {} found {} zero timesteps: ".format(rank, len(zero_filter)) + str(zero_filter))
    for index in indexes:
        slice_x, slice_y = index_to_chunk(index, chunk_x, data_x, trim_x, chunk_y, data_y, trim_y, True)
        #                            NB switched order below
        try:
            ts_data = imstack.slice2cube(slice_x, slice_y, correct=opts.pbcor)
        except ZeroDivisionError:
            ts_data = np.nan*np.ones((chunk_y, chunk_x, 20))
        if opts.remove_zeros:
            ts_data = np.delete(ts_data, zero_filter, axis=2)
        # mean
        data[..., 0] = np.average(ts_data, axis=2)
        if N_MOMENTS > 2:
            if opts.filter_lo:
                ts_data = filtfilt(bb, ab, ts_data, axis=2)
            data[..., 2] = skew(ts_data, axis=2)
        if N_MOMENTS > 3:
            data[..., 3] = kurtosis(ts_data, axis=2)
        if N_MOMENTS > 1:
            if N_MOMENTS == 1:
                if opts.filter_lo:
                    ts_data = filtfilt(bb, ab, ts_data, axis=2)
            if opts.filter_hi:
                ts_data = filtfilt(bb_hi, ab_hi, ts_data, axis=2)
            data[..., 1] = np.std(ts_data, axis=2)
        comm.send(data, dest=0, tag=index)
    print("Worker rank {} done".format(rank))
