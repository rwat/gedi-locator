import numpy as np
import unittest

from gedi_locator.polyder import polyder


class PolyderTest(unittest.TestCase):

    def test_polyder_basic(self):
        expected_result = np.array([2.0, 1.0])
        actual_result = polyder(np.array([1.0, 1.0, 11.0]), 1)
        self.assertTrue((expected_result == actual_result).all())

    
    def test_polyder_compare_to_numpy(self):
        polynomial = np.array([-3.255876851837828e-11, -4.181289098687833e-09,
                               1.053247616259639e-06, -8.122534319705013e-05,
                               -0.014598707999556891, 6.129419891206174,
                               -787.6348969694789, -198346.9972731084])
        local_result = polyder(polynomial, 1)
        np_result = np.polyder(polynomial, 1)
        self.assertTrue((local_result == np_result).all())