import numpy as np
import unittest

from shared import polyval, is_overlap_sorted_values, bbox_intersect,
    _deg_to_rad, _rad_to_deg, _angle_from_slope,
    _one_degree_latitude_distance, _slope_pos_vert_distance,
    _slope_neg_vert_distance, poly_bbox

class PolyvalTest(unittest.TestCase):

    def test_polyval_basic(self):
        expected_result = 16.0
        actual_result = polyval(np.array([2.0, 1.0, 11.0]))
        self.assertTrue((expected_result == actual_result).all())
    
    def test_polyval_estimate(self):
        # first partition of 'GEDI01_B_2019121002259_O02161_T03591_02_003_01.json'
        # {'avg_error': 0.007471593775443627, 'lat_max': -6.735598138002143, 'lat_min': -9.603002476444333, 'left_extent': 0, 'lon_max': -179.76997215340455, 'lon_min': -179.9998966712299, 'max_error': 0.02993653485339064, 'pn': [-3.255876851837828e-11, -4.181289098687833e-09, 1.053247616259639e-06, -8.122534319705013e-05, -0.014598707999556891, 6.129419891206174, -787.6348969694789, -198346.9972731084], 'right_extent': 1497}
        lon_range = [-179.9998966712299, -179.76997215340455]
        max_error = 0.02993653485339064
        #lat_range = [-9.603002476444333, -6.735598138002143]
        polynomial = np.array([-3.255876851837828e-11, -4.181289098687833e-09,
                               1.053247616259639e-06, -8.122534319705013e-05,
                               -0.014598707999556891, 6.129419891206174,
                               -787.6348969694789, -198346.9972731084])
                        
        x = polyval(polynomial, -179.76997215340455)
        #x = -9.214415923081106





