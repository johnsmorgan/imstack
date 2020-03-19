import pytest
import numpy as np
from image_stack import *

a = np.array([[1., 1., 2., 2.], [1., 1., 1., 1.], [2., 2., 2., 2.]])
print(hypotn(a))

def test_hypotn():
    assert (np.isclose(hypotn(a), np.array([[2.44948974, 2.44948974, 3., 3.]]))).all()
