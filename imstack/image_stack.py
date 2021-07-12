import numpy as np
from math import sqrt, floor, ceil, pi, hypot
import h5py
from astropy import wcs

def pix2stamp(x, y, n):
    """
    produce slices for a square centred on x, y.
    x, y are real numbers
    return x_slice, y_slice

    >>> pix2stamp(1.6, 3.2, 1)
    (slice(2, 3, None), slice(3, 4, None))
    >>> pix2stamp(1.6, 3.2, 2)
    (slice(1, 3, None), slice(3, 5, None))
    >>> pix2stamp(1.6, 3.2, 3)
    (slice(1, 4, None), slice(2, 5, None))
    """
    if n % 2:
        return tuple(slice(int(round(i)) - (n-1)/2, int(round(i)) + (n+1)/2) for i in (x, y))
    else:
        return tuple(slice(int(ceil(i)) - n/2, int(ceil(i)) + n/2) for i in (x, y))

def semihex(data, axis=None):
    """
    Calculate standard deviation via semi-interhexile range.
    """
    h1, h5 = np.percentile(data, (100/6., 500/6.), axis=axis)
    return (h5-h1)/2.

def hypotn(d, axis=0):
    """
    similar to hypot but not restricted to 2
    """
    return np.sqrt(np.sum(d**2, axis=axis))

def sault_beam(beam, sigma):
    """
    combine beams with sault weighting
    """
    return (np.sum(beam**2/sigma**2, axis=0)/np.sum(sigma**-2))**0.5

def sault_weight(data, beam, sigma, correct=False):
    """
    For combining data from multiple pointings / polarisations

    See 1996A&AS..120..375S Equation 1.
    Pointing / Polarisation must be first axis for both data and beam
    """
    while len(data.shape) > len(beam.shape):
        beam = beam[..., np.newaxis]
    while len(data.shape) > len(sigma.shape):
        sigma = sigma[..., np.newaxis]
    d = np.sum(data*beam/sigma**2, axis=0)/np.sum(beam**2/sigma**2, axis=0)
    if correct is True:
        return d
    return d*sault_beam(beam, sigma)

class ImageStack(object):
    def __init__(self, h5filename, image_type='image', freq=None, steps=None, mode='r'):
        self.df = h5py.File(h5filename, mode)
        assert image_type in ('image', 'dirty', 'model', 'moment'), "Unsupported image_type %s." % image_type
        #assert self.df.attrs['VERSION'] == '0.1', "Wrong version. Expected 0.1, got %s" % self.df.attrs['VERSION']
        self.image_type = image_type
        if freq is None:
            self.group = self.df['/']
        else:
            self.group = self.df[freq]
        self.data = self.group[image_type]
        self.header = self.group['header'].attrs
        self.wcs = wcs.WCS({k: v.decode('ascii') if isinstance(v, bytes) else v for k, v in self.header.items()})
        self.channel = 0
        if steps is None:
            self.steps = [0, self.data.shape[-1]]
        else:
            self.steps = [steps[0], steps[1]]
            if self.steps[1] is None:
                self.steps[1] = self.data.shape[-1]
        if 'SIGMA' in self.group['beam'].attrs:
            self.sigma = self.group['beam'].attrs['SIGMA']
        else:
            self.sigma = np.array((1.0, 1.0))
        print(f"sigma={self.sigma})")
        if 'SCALE' in self.group['beam'].attrs:
            self.scale = self.group['beam'].attrs['SCALE']
        else:
            self.scale = np.array([1.0])
        print(f"scale={self.scale})")

    def update(self, freq=None, image_type=None, steps=None):
        if freq is not None:
            self.group = self.df[freq]
            self.data = self.group[self.image_type]
            self.header = self.group['header'].attrs
            self.wcs = wcs.WCS(self.header)

        if image_type is not None:
            assert image_type in ('image', 'dirty', 'model', 'moment'), "Unsupported image_type %s." % image_type
            self.image_type = image_type
            self.data = self.group[image_type]

        if steps is not None:
            self.steps = [steps[0], steps[1]]
            if self.steps[1] is None:
                self.steps[1] = self.data.shape[-1]

    def check_slice(self, x, y, margin=0):
        """
        check slice is valid (not off the edge of the image)
        """
        sqrt_beamsize = sqrt(self.get_pixel_beam())
        x_slice, y_slice = pix2stamp(x, y, margin+int(round(10*max(1, sqrt_beamsize))))
        if not all(0 < s < self.data.shape[2] for s in (x_slice.start, x_slice.stop)):
            return 1
        if not all(0 < s < self.data.shape[1] for s in (y_slice.start, y_slice.stop)):
            return 1
        return 0

    def get_interval(self):
        """
        Get interval between neighbouring images
        """
        return self.group.attrs['TIME_INTERVAL']

    def get_intervals(self):
        """
        Get series of intervals
        """
        return self.group.attrs['TIME_INTERVAL']*np.arange(self.steps[1]-self.steps[0])

    def world2pix(self, ra, dec, floor=True):
        """
        return pixel coordinates x, y
        NB x is the fastest varying axis!
        """
        pixcoord = self.wcs.celestial.wcs_world2pix(np.array([[ra, dec]]), 0)
        if floor:
            pixcoord = np.round(pixcoord).astype(np.int)
        return pixcoord[0, 0], pixcoord[0, 1]

    def get_pixel_beam(self):
        """
        Get size of *synthesised* beam in pixels.
        NB assumes square pixels and roughly circular synth beam
        """
        return abs(pi*0.25*self.header['BMAJ']*self.header['BMIN']/(self.header['CDELT1']*self.header['CDELT2']))
    
    def get_scale(self):
        if 'SCALE' in self.group['beam'].attrs:
            return self.group['beam'].attrs['SCALE']
        return 1.0

    def scale_beam(self, beam):
            if len(beam.shape) == 1:
                return beam * self.get_scale()[:, 0, 0, self.channel, 0]
            elif len(beam.shape) == 3:
                return beam * self.get_scale()[..., self.channel, 0]
            else:
                raise RuntimeError("don't know how to deal with beam shape %s" % (beam.shape))

    def pix2beam(self, x, y, avg_pol=True, scale=True):
        """
        get beam corresponding to x,y
        """
        beam = self.group['beam'][:, y, x, self.channel, 0]
        if scale == True:
            beam = self.scale_beam(beam)
        if avg_pol is True:
            if not np.any(beam):
                return 0.0
            return sault_beam(beam, self.sigma)
        else:
            return beam

    def world2beam(self, ra, dec, avg_pol=True, scale=True):
        """
        get beam corresponding to ra,dec
        """
        x, y = self.world2pix(ra, dec)
        return self.pix2beam(x, y, avg_pol, scale=scale)

    def pix2ts(self, x, y, avg_pol=True, correct=True):
        """
        Produce a timeseries for a given RA and Decl.
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        ts = self.data[:, y, x, self.channel, self.steps[0]:self.steps[1]].astype(np.float_)
        if avg_pol is True:
            beam = self.pix2beam(x, y, False)
            return sault_weight(ts, beam, self.sigma, correct)
        else:
            if correct is True:
                beam = self.pix2beam(x, y, False)
                return ts/beam[:, np.newaxis]
            return ts

    def world2ts(self, ra, dec, avg_pol=True, correct=True):
        """
        Produce a timeseries for a given RA and Decl.
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        x, y = self.world2pix(ra, dec)
        return self.pix2ts(x, y, avg_pol, correct)

    def slice2cube(self, x_slice, y_slice, avg_pol=True, correct=True):
        """
        Produce an nxm cube centred
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        ts = self.data[:, y_slice, x_slice, self.channel, self.steps[0]:self.steps[1]].astype(np.float_)
        beam = self.pix2beam(x_slice, y_slice, False)
        if avg_pol is True:
            return sault_weight(ts, beam, self.sigma, correct)
        else:
            if correct is True:
                return ts/beam
            return ts

    def pix2cube(self, x, y, n, m=None, avg_pol=True, correct=True):
        """
        Produce an nxm cube centred on a given RA and Decl.
        n is required
        m defaults to n
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        if m is None:
            m = n
        x_slice, y_slice = pix2stamp(x, y, n)
        return self.slice2cube(x_slice, y_slice, avg_pol, correct)

    def pix2rms(self, x, y, avg_pol=True, correct=True):
        """
        Calculate local rms centred on x,y for each timestep
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        cube = self.pix2cube(x, y, int(round(10*max(1, sqrt_beamsize))), avg_pol=avg_pol, correct=correct)
        if avg_pol is True:
            #cube has axes x, y, time
            return semihex(cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2]), axis=0)
        else:
            #cube has axes pol, x, y, time
            return semihex(cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2], cube.shape[3]), axis=1)

    def get_continuum(self, avg_pol=True, correct=True, sigma_weight=True, scale=True):
        """
        Generate continuum image.
        """

        self.header = self.group['header'].attrs
        cont = self.group['continuum'][:, :, :, self.channel, 0]
        beam = self.group['beam'][:, :, :, self.channel, 0]
        if scale == True:
            beam = self.scale_beam(beam)
        if sigma_weight is True:
            sigma = self.sigma
        else:
            sigma = np.array((1.0, 1.0))
        if avg_pol is True:
            return sault_weight(cont, beam, sigma, correct)
        else:
            if correct is True:
                return cont/beam
            return cont

if __name__ == "__main__":
    """
    run tests
    """
    import doctest
    doctest.testmod()
