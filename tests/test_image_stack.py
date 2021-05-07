import pytest
import numpy as np
from imstack.image_stack import *

np.set_printoptions(precision=16)
a = np.array([[1., 1., 2., 2.], [1., 1., 1., 1.], [2., 2., 2., 2.]])

def test_hypotn():
    np.testing.assert_array_almost_equal_nulp(hypotn(a), np.array([2.449489742783178, 2.449489742783178, 3., 3.]))

def test_sault_weight():
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((1., 1.)), beam=np.array((1., 1.)), sigma=np.array((1., 1.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((1., 1.)), beam=np.array((1., 1.)), sigma=np.array((1., 1.)), correct=True), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((1., 1.)), beam=np.array((0.5, 0.5)), sigma=np.array((1., 1.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((2., 1.)), beam=np.array((0.5, 0.5)), sigma=np.array((1., 1.))), 1.5)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((2., 1.)), beam=np.array((0.5, 0.5)), sigma=np.array((2., 1.))), 1.2)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((2., 1.)), beam=np.array((0.5, 0.5)), sigma=np.array((4., 2.))), 1.2)
     np.testing.assert_array_almost_equal_nulp(sault_weight(data=np.array((1., 1.)), beam=np.array((0.5, 0.5)), sigma=np.array((1., 1.)), correct=True), 2.0)

def test_sault_beam():
     np.testing.assert_array_almost_equal_nulp(sault_beam(beam=np.array((1., 1.)), sigma=np.array((1., 1.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_beam(beam=np.array((1., 1.)), sigma=np.array((1., 5.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_beam(beam=np.array((0.5, 0.25)), sigma=np.array((1., 1.))), 0.3952847075210474)
     np.testing.assert_array_almost_equal_nulp(sault_beam(beam=np.array((0.5, 0.25)), sigma=np.array((2., 1.))), 0.31622776601683794)
     
