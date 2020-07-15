import math
import os
import numpy as np
from numba import jit

from polyder import polyder

# one degree latitude distance in kilometers
ODL_DISTANCE = 110.5743
# Factor in error introduced by spherical trigonometry
STATIC_MULT = 1.003
# Distance from track 6 to 0 perpendicular to orbit
PERPENDICULAR_ABOVE_DISTANCE = 3.0125  # in km
# Distance from track 6 to 7 perpendicular to orbit
PERPENDICULAR_BELOW_DISTANCE = 1.2125  # in km


def get_filenames(path):
    """ Takes a filesystem path and returns a sorted list of filenames under
        that path. """
    xs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        xs.extend(filenames)
        break
    xs.sort()
    return xs


@jit
def polyval(p, x):
    """ Takes a sequence p representing a polynomial and a number x and
        returns the value of p at x. This version is Numba-compatible; NumPy's
        version is not. """
    val = 0
    ii = len(p) - 1
    for i in range(len(p) - 1):
        val += p[i] * (x ** ii)
        ii -= 1
    return val + p[-1]


@jit
def is_overlap_sorted_values(v1, v2, w1, w2):
    """ Takes two pairs of values, v1, v2 and w1, w2 and returns a boolean
        result indicating whether the range v1, v2 (inclusive) contains any
        values in the range w1, w2 (inclusive). """
    if (v2 < w1) or (v1 > w2):
        return False
    else:
        return True


@jit
def bbox_intersect(a_ary, b_ary):
    """ Takes two nested two dimensional arrays, a_ary and b_ary,
        representing a bounding box in ll, ur format
        [[lon, lat], [lon, lat]]. Returns a boolean result as to whether the
        bounding boxes overlap. """
    # Do any of the 4 corners of one bbox lie inside the other bbox?
    # bbox format of [ll, ur]
    # bbx[0] is lower left
    # bbx[1] is upper right
    # bbx[0][0] is lower left longitude
    # bbx[0][1] is lower left latitude
    # bbx[1][0] is upper right longitude
    # bbx[1][1] is upper right latitude

    # Detect longitude and latitude overlap
    if is_overlap_sorted_values(a_ary[0][0], a_ary[1][0], b_ary[0][0], b_ary[1][0]) \
            and is_overlap_sorted_values(a_ary[0][1], a_ary[1][1], b_ary[0][1], b_ary[1][1]):
        return True
    else:
        return False


@jit
def _deg_to_rad(deg):
    """ Takes a degree value deg and returns the equivalent value in
        radians. """
    return deg * math.pi / 180


@jit
def _rad_to_deg(rad):
    """ Takes a radian value rad and returns the equivalent value in
        degrees. """
    return rad * 180 / math.pi


@jit
def _angle_from_slope(poly_ary, lon):
    """ Takes a polynomial array poly_ary and a longitude value lon and
        returns the angle in radians at that longitude. """
    # find the first derivative
    poly = polyder(poly_ary, 1)
    # get the slope at our point of interest
    slope = polyval(poly, lon)
    # get the angle from the slope
    angle = math.atan(slope)
    return angle  # in radians


@jit
def _slope_pos_vert_distance(B, perp):
    """ Takes an acute angle B and distance in kilometers perp and returns the
        distance perpendicular to the equator. Assumes an orbital section
        positive in slope. """
    # get the arc length of 'a' at around this latitude
    a = _deg_to_rad(perp / ODL_DISTANCE)
    # arclength from beam 0110 to first or last beam vertically
    arclength = math.atan(math.tan(a) / math.cos(B))
    # distance in km from beam 0110 to first or last vertically
    beam_distance = _rad_to_deg(arclength) * ODL_DISTANCE
    return beam_distance  # in km


@jit
def _slope_neg_vert_distance(B, perp):
    """ Takes an acute angle B and distance in kilometers perp and returns the
        distance perpendicular to the equator. Assumes an orbital section
        negative in slope. """
    # get the arc length of 'a' at around this latitude
    c = _deg_to_rad(perp / ODL_DISTANCE)
    # arclength from beam 0110 to first or last beam vertically
    arclength = math.atan(math.cos(B) * math.tan(c))
    # distance in km from beam 0110 to first or last vertically.
    beam_distance = _rad_to_deg(arclength) * ODL_DISTANCE
    return beam_distance  # in km


@jit
def poly_bbox(poly_ary, lon_min, lon_max, max_error):
    """ Takes a polynomial array poly_ary, starting longitude lon_min, ending
        longitude lon_max, and error distance max_error and returns a nested
        array representing a bounding box for a GEDI swath extent. Return
        value is in ll, ur format [[lon, lat], [lon, lat]]. """
    # Angle 'B' is also theta
    B_lon_min = _angle_from_slope(poly_ary, lon_min)
    B_lon_max = _angle_from_slope(poly_ary, lon_max)

    # Slope assumptions:
    # 1) will never be 0 at orbit major maximum or minimum
    # 2) will never change sign because of #1
    # Therefore, slopes are assumed continuously increasing or decreasing
    slope_is_positive = True if B_lon_min > 0 else False

    if slope_is_positive:
        # lat min will be at min lon
        lat_min_ = polyval(poly_ary, lon_min)
        # lat max will be at max lon
        lat_max_ = polyval(poly_ary, lon_max)

        # find the latitudinal distance using rules for right spherical triangles
        distance_below = _slope_pos_vert_distance(B_lon_min, PERPENDICULAR_BELOW_DISTANCE)
        distance_above = _slope_pos_vert_distance(B_lon_max, PERPENDICULAR_ABOVE_DISTANCE)

    else:  # slope is negative
        # lat min will be at max lon
        lat_min_ = polyval(poly_ary, lon_max)
        # lat max will be at min lon
        lat_max_ = polyval(poly_ary, lon_min)
        
        # find the latitudinal distance using rules for right spherical triangles
        distance_below = _slope_neg_vert_distance(B_lon_min, PERPENDICULAR_BELOW_DISTANCE)
        distance_above = _slope_neg_vert_distance(B_lon_max, PERPENDICULAR_ABOVE_DISTANCE)

    # subtract distance from 0110 beam to 0111 beam
    lat_min = lat_min_ - ((distance_below * STATIC_MULT) - max_error) / ODL_DISTANCE
    # add distance from 0110 beam to 0000 beam
    lat_max = lat_max_ + ((distance_above * STATIC_MULT) + max_error) / ODL_DISTANCE
    
    # bbox format of [ll, ur]
    bbox = np.array([[lon_min, lat_min], [lon_max, lat_max]])

    return bbox
