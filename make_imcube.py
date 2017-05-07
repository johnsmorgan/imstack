import os, datetime
import numpy as np
from h5pycache import File
from astropy.io import fits

#HDF_NAME = "1114303032.hdf5"
HDF_NAME = "1114303032.hdf5"
VERSION = "0.1"
CACHE_SIZE=24
N=2
TIME_INTERVAL=0.5
TIME_INDEX=1
HEADER_INDEX=100
POLS = ['XX', 'YY']
FREQS = ["055-066", "118-129"]
FREQS = ["118-129"]
STAMP_SIZE=16
PB_FILE="../continuum/{freq}-{pol}-beam.fits"
SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]
HDU = 0
PB_THRESHOLD = None # fraction of pbmax
#SUFFIXES=["image", "model", "dirty"]
SUFFIXES=["image"]
N_TIMESTEPS=591
N_CHANNELS=1
#DTYPE = np.float32
DTYPE = np.float16
FILENAME="../images/{freq}-t{time:04d}-{pol}-{suffix}.fits"

# FIXME include a version

if os.path.exists(HDF_NAME):
    print "Warning: editing existing file"
    file_mode = "r+"
else:
    file_mode = "w"

for suffix in SUFFIXES:
    for freq in FREQS:
        for t in xrange(TIME_INDEX, N_TIMESTEPS+TIME_INDEX):
            for p in POLS:
                infile = FILENAME.format(freq=freq, time=t, pol=p, suffix=suffix)
                if not os.path.exists(infile):
                    raise IOError, "couldn't find file %s" % infile

with File(HDF_NAME, file_mode, 0.9*CACHE_SIZE*1024**3, 1) as df:
    df.attrs['VERSION'] = VERSION
    df.attrs['USER'] = os.environ['USER']
    df.attrs['DATE_CREATED'] = datetime.datetime.utcnow().isoformat()
    
    for freq in FREQS:
        if not freq in df.keys():
            group = df.create_group(freq)
        else:
            print "Warning, overwriting existing frequency %s" % freq
            group = df[freq]
        group.attrs['TIME_INTERVAL'] = TIME_INTERVAL

        # determine data size and structure 
        image_file = FILENAME.format(freq=freq, time=TIME_INDEX, pol=POLS[0], suffix=SUFFIXES[0])
        hdus = fits.open(image_file, memmap=True)
        image_size = hdus[HDU].data.shape[-1]
        assert image_size % STAMP_SIZE == 0, "image_size must be divisible by STAMP_SIZE"
        data_shape = [len(POLS), image_size, image_size, N_CHANNELS, N_TIMESTEPS]
        chunks = (len(POLS), STAMP_SIZE, STAMP_SIZE, N_CHANNELS, N_TIMESTEPS)

        beam_shape = data_shape[:-1] + [1] # just one beam for all timesteps for now
        beam = group.create_dataset("beam", beam_shape, dtype=np.float32, compression='gzip', shuffle=True)
        for p, pol in enumerate(POLS):
            hdus = fits.open(PB_FILE.format(freq=freq, pol=pol), memmap=True)
            beam[p, :, :, 0, 0] = hdus[HDU].data[SLICE]
            for key, item in hdus[0].header.iteritems():
                beam.attrs[key] = item

        # write main header information
        timesteps = group.create_dataset("WSCTIMES", (N_TIMESTEPS,), dtype=np.uint16)
        timesteps2 = group.create_dataset("WSCTIMEE", (N_TIMESTEPS,), dtype=np.uint16)
        header_file = FILENAME.format(freq=freq, time=HEADER_INDEX, pol=POLS[0], suffix=SUFFIXES[0])
        # add fits header to attributes
        hdus = fits.open(header_file, memmap=True)
        header = group.create_dataset('header', data=[], dtype=DTYPE)
        for key, item in hdus[0].header.iteritems():
            header.attrs[key] = item

        for s, suffix in enumerate(SUFFIXES):
            # gzip is rather slower than lzf, but is more standard in hdf5. Will allow dumping to h5dump etc.
            data = group.create_dataset(suffix, data_shape, chunks=chunks, dtype=DTYPE, compression='gzip', shuffle=True)
            filenames = group.create_dataset("%s_filenames" % suffix, (len(POLS), N_CHANNELS, N_TIMESTEPS), dtype="S%d" % len(header_file.format(freq=freq, suffix=suffix)), compression='gzip')
        
            n_rows = image_size/N
            for i in range(N):
                print "processing segment %d/%d" % (i+1, N)
                for t in xrange(N_TIMESTEPS):
                    im_slice = [slice(n_rows*i, n_rows*(i+1)), slice(None, None, None)]
                    if len(SLICE) > 2:
                        fits_slice = SLICE[:-2] + im_slice
                    else:
                        fits_slice = im_slice

                    for p, pol in enumerate(POLS):
                        infile = FILENAME.format(freq=freq, time=t+TIME_INDEX, pol=pol, suffix=suffix)
                        print " processing %s" % (infile)
                        hdus = fits.open(infile, memmap=True)
                        filenames[p, 0, t] = infile
                        data[p, n_rows*i:n_rows*(i+1), :, 0, t] = hdus[0].data[fits_slice]
                        if s==0 and p==0:
                            timesteps[t] = hdus[0].header['WSCTIMES']
                            timesteps2[t] = hdus[0].header['WSCTIMEE']
                        else:
                            # NB these are *not* enforced across different frequency bands, but these could, in principle, have different TIME_INTERVALS
                            assert timesteps[t] == hdus[0].header['WSCTIMES'], "Timesteps do not match %s in %s" % (SUFFIXES[0], infile)
                            assert timesteps2[t] == hdus[0].header['WSCTIMEE'], "Timesteps do not match %s in %s" % (SUFFIXES[0], infile)
