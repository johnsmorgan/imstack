import pytest
import numpy as np
from imstack.image_stack import *

np.set_printoptions(precision=16)
a = np.array([[1., 1., 2., 2.], [1., 1., 1., 1.], [2., 2., 2., 2.]])
print(hypotn(a))

def test_hypotn():
    np.testing.assert_array_almost_equal_nulp(hypotn(a), np.array([2.449489742783178, 2.449489742783178, 3., 3.]))

def test_sault_weight():
     np.testing.assert_array_almost_equal_nulp(sault_weight(np.array((1., 1.)), np.array((1., 1.)), np.array((1., 1.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(np.array((1., 1.)), np.array((1., 1.)), np.array((1., 1.)), correct=True), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(np.array((1., 1.)), np.array((0.5, 0.5)), np.array((1., 1.))), 1.0)
     np.testing.assert_array_almost_equal_nulp(sault_weight(np.array((1., 1.)), np.array((0.5, 0.5)), np.array((1., 1.)), correct=True), 2.0)
