from bisect import bisect
from copy import deepcopy
from geopy.distance import geodesic
import json
import math
from multiprocessing import Pool
import os
import numpy as np
from scipy.optimize import minimize
from statistics import mean
import time
import warnings

import settings
from shared import poly_bbox

INPUT_PATH = settings.COORDS_PATH
OUTPUT_PATH = settings.PARTITIONS_30M_PATH
ERROR_THRESHOLD = 0.030  # in kilometers


def get_filenames(path):
    xs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        xs.extend(filenames)
        break
    xs.sort()
    return xs


def split_on_anti_meridian(lons, lats, pn_degree):
    """ Take two lists of lists, lons and lats, and return two lists of lists
        with lons and lats split on the antimeridian. """
    # Working across the anti meridian boundary is numerically challenging so
    #  it's best to break up our data along that boundary.

    def _fn(lons, lats, pn_degree):
        am = 0
        for i in range(1, len(lons)):
            if (lons[i-1] > 0) and (lons[i] < 0):
                am = i
                break
        if am > 0:
            lons_ = []
            lats_ = []
            if am > pn_degree:
                lons_.append(lons[0:am])
                lats_.append(lats[0:am])
            if (len(lons) - am) > pn_degree:
                lons_.append(lons[am:])
                lats_.append(lats[am:])
            return lons_, lats_
        else:
            return [lons], [lats]

    acc_lons = []
    acc_lats = []
    for lons_, lats_ in zip(lons, lats):
        lons_parts, lats_parts = _fn(lons_, lats_, pn_degree)
        acc_lons.extend(lons_parts)
        acc_lats.extend(lats_parts)
    
    return acc_lons, acc_lats


def _split_on_inflections(lons, lats, pn_degree):
    # A low degree polynomial will be less prone to fitting local maxima.
    # We're optimizing over a path which is convex and smooth.
    inf_pn_degree = 3
    lats_peak_threshold = 51.4
    skip_amount = 250000
    inflection_threshold = 1e-6
    abs_lats = np.abs(lats)
    equatorial_markers = []
    inflection_indices = []

    # assemble a list of equatorial markers 
    search = True
    left_boundary = 0
    while search:
        search = False
        for i in range(left_boundary, len(abs_lats)):
            if 0 < abs_lats[i] < 1:
                equatorial_markers.append(i)
                left_boundary = i + skip_amount
                if left_boundary < len(abs_lats):
                    search = True
                break

    # add beginning and end indices to equatorial markers
    markers = [0, *equatorial_markers, len(lons) - 1]

    # Determine if there are any inflection points
    for marker_index in range(1, len(markers)):
        left_boundary = markers[marker_index-1]
        right_boundary = markers[marker_index]
        
        # Determine possible inflection range
        peak_indices = None
        for i in range(left_boundary, right_boundary):
            if abs_lats[i] > lats_peak_threshold:
                for ii in range(i, right_boundary):
                    if abs_lats[ii] < lats_peak_threshold:
                        peak_indices = [i, ii]
                        break
                if not peak_indices:
                    peak_indices = [i, right_boundary - 1]
                break
        
        if peak_indices:
            x1, x2 = peak_indices
            # Fit a function and test for inflection point
            pn = generate_polynomial(inf_pn_degree, lons[x1:x2], abs_lats[x1:x2])
            # Need to take the inverse of the fitted function because
            #   scipy.optimize doesn't have 'maximize'
            pn_inverse = np.poly1d([x * -1 for x in pn])
            opt = minimize(pn_inverse, x0=lons[x1], bounds=[[lons[x1], lons[x2]]])
            if opt.success:
                lon_min = opt.x[0]
                z = inflection_threshold
                # If polynomial max value isn't located at either end of our range then
                #   an inflection point has been found
                if (abs(lon_min - lons[x1]) > z) and (abs(lon_min - lons[x2]) > z):
                    # find the offset value based on inflection point
                    inflection_indices.append(bisect(lons, lon_min))
    
    # split up lons and lats according to inflection_indices
    if inflection_indices:
        final_indices_ = [0, *inflection_indices, len(lons) - 1]
        final_indices = []
        # Discard split at inflection if it is too small for us to fit a
        #   polynomial function
        for i in range(1, len(final_indices_)):
            if (final_indices_[i] - final_indices_[i-1]) > pn_degree:
                final_indices.append(final_indices_[i-1])
        final_indices.append(final_indices_[-1])

        result_lons = []
        result_lats = []
        for i in range(1, len(final_indices)):
            left_boundary = final_indices[i-1]
            right_boundary = final_indices[i]
            result_lons.append(lons[left_boundary:right_boundary])
            result_lats.append(lats[left_boundary:right_boundary])
        return result_lons, result_lats
    else:
        return [lons], [lats]


def split_on_inflections(lons, lats, pn_degree):
    """ Find the inflections through the anti-meridian halves of a GEDI
        orbit. Takes an array of arrays of longitude values lons, and array
        of arrays of latitude values lats, and the degree of polynomial to
        use to find inflections and returns an array of arrays of longitude
        and latitude values. """
    acc_lons = []
    acc_lats = []
    for lons_, lats_ in zip(lons, lats):
        lons_parts, lats_parts = _split_on_inflections(lons_, lats_, pn_degree)
        acc_lons.extend(lons_parts)
        acc_lats.extend(lats_parts)
    return acc_lons, acc_lats


def generate_polynomial(deg, lons, lats):
    """ Takes degree of polynomial deg, array of longitude values lons, and
        array of latitude values lats and attempts to fit a polynomial to the
        data via non-linear regression (least squares).  Returns the
        polynomial."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        pf = np.polyfit(lons, lats, deg)
        return np.poly1d(pf)


def get_fn_error_rates(fn, lons, lats):
    """ Take a function fn, array of longitude values lons, and array of
        latitude values lats and return the mean average error and maximum
        error between function-calculated latitude and reference latitude
        values. """
    # Because we're computing distance between latitudes on the same longitude
    #   the longitudinal value may be any value; 0, in this case
    # One degree latitude is 110.567 km at the equator and 111.699 at the poles.
    #   We could have used the Haversine formula which assumes a Great-circle but
    #   the geodetic distance uses an ellipsoid representation and is more
    #   accurate (WGS-84 ellipsoid by default)
    deltas_lat = [((lats[i], 0), (fn(lons[i]), 0)) for i in range(len(lons))]
    deltas_km = [geodesic(*points).kilometers for points in deltas_lat]
    return mean(deltas_km), max(deltas_km)


def apply_partition_latitudes(p):
    """ Takes a dict p representing partition data and mutates that partition
        data to include its minimum and maximum latitude values. """
    bbox = poly_bbox(p['pn'], p['lon_min'], p['lon_max'], p['max_error'])
    lat_range = {'lat_min': bbox[0][1], 'lat_max': bbox[1][1]}
    return {**p, **lat_range}


def create_dynamic_partitions(lons__, lats__):
    max_inc_idx = 0
    max_increments = [100, 1000, 20000]
    pn_degree = 7
    error_threshold = ERROR_THRESHOLD
    results = []
    partition_num = 1

    lons_, lats_ = split_on_anti_meridian([lons__], [lats__], pn_degree)
    lons_, lats_ = split_on_inflections(lons_, lats_, pn_degree)

    for s_lons, s_lats in zip(lons_, lats_):
        lower_bound = 0
        upper_bound = pn_degree + 1
        known_good = pn_degree + 1
        known_bad = len(s_lons)

        while (upper_bound + 1) < len(s_lons):
            while True:
                lons = s_lons[lower_bound:upper_bound]
                lats = s_lats[lower_bound:upper_bound]
                pn = generate_polynomial(pn_degree, lons, lats)
                avg_error, max_error = get_fn_error_rates(pn, lons, lats)
                
                print(f'p:{partition_num}, lower_bound:{lower_bound}, known_good:{known_good}, upper_bound:{upper_bound}, known_bad:{known_bad}, max_error:{max_error}')

                # Partition width must be at least one more than the
                #   degree of polynomial
                if ((known_bad - known_good) == 1) or (known_good > known_bad):
                    upper_bound = known_good
                    lons = s_lons[lower_bound:upper_bound]
                    lats = s_lats[lower_bound:upper_bound]
                    pn = generate_polynomial(pn_degree, lons, lats)
                    avg_error, max_error = get_fn_error_rates(pn, lons, lats)
                    break
                
                # the partition may grow
                if max_error <= error_threshold:
                    known_good = upper_bound
                    maybe_increment = int((known_bad - upper_bound) / 2)
                    if maybe_increment > max_increments[max_inc_idx]:
                        increment = max_increments[max_inc_idx]
                        max_inc_idx += 1
                        if max_inc_idx >= len(max_increments):
                            max_inc_idx = len(max_increments) - 1
                    else:
                        increment = maybe_increment
                    upper_bound = upper_bound + increment
                    if upper_bound == known_good:
                        upper_bound += 1
                # the partition must shrink
                else:
                    known_bad = upper_bound
                    upper_bound = upper_bound - int((known_bad - known_good) / 2)
                    if upper_bound == known_bad:
                        upper_bound -= 1
                    max_inc_idx = 0
            
            lon_min = s_lons[lower_bound]
            lon_max = s_lons[upper_bound]
            
            partition = {'left_extent': lower_bound,
                         'right_extent': upper_bound,
                         'pn': tuple(pn.c),
                         'lon_min': lon_min,
                         'lon_max': lon_max,
                         'max_error': max_error,
                         'avg_error': avg_error}

            # compute bounding box for current partition
            partition = apply_partition_latitudes(partition)

            results.append(partition)

            lower_bound = upper_bound + 1
            upper_bound = lower_bound + pn_degree + 1
            known_good = lower_bound + pn_degree + 1
            known_bad = len(s_lons)

            partition_num += 1

    # Sort partitions by longitude. This will introduce a gap in the data if
    #   the orbit passes through the anti-meridian (which happens most of the
    #   time). It's not a problem for the search but it's worth knowing it's
    #   there. Makes the data binary search compatible on a per granule or
    #   orbit basis.
    return sorted(results, key=lambda x: x['lon_min'])


def filter_invalid_coords(lons_, lats_):
    """ Takes an array of longitude values lons_ and an array of latitude
        values lats_ and returns a copy of each with invalid with invalid
        point data removed. """
    lons = deepcopy(lons_)
    lats = deepcopy(lats_)
    for i in range(len(lons)-1, 0, -1):
        if (lons[i] < -180) or (lons[i] > 180) or math.isnan(lons[i]) \
          or (lats[i] < -90) or (lats[i] > 90) or math.isnan(lats[i]):
            print(f'Bad data removed at index {i}: {lons[i]}, {lats[i]}')
            del lons[i]
            del lats[i]
    return lons, lats


def do(input_filenames, input_path, output_path):
    for input_filename in input_filenames:
        print('\n' + input_filename)
        with open(input_path + input_filename) as coords:

            coords_json = json.load(coords)

            # Check for valid data; without this check I was receiving the
            #   following error on some files: "ValueError: On entry to DLASCL
            #   parameter number 4 had an illegal value". Turns out there are
            #   some NaN's in GEDI coordinate data.
            lons, lats = filter_invalid_coords(coords_json['lons'],
                                               coords_json['lats'])

            partitions = create_dynamic_partitions(lons, lats)

            with open(output_path + input_filename + '.json', 'w') as output_file: 
                json.dump(partitions, output_file)


if __name__ == '__main__':
    start_time = time.time()
    
    #with Pool(multiprocessing.cpu_count()) as p:
    #   p.starmap(do, [[[x], INPUT_PATH, OUTPUT_PATH] for x in (get_filenames(INPUT_PATH))])

    #do((get_filenames(INPUT_PATH))[8:9], INPUT_PATH, OUTPUT_PATH)
    do(['GEDI01_B_2019108002011_O01959_T03909_02_003_01'], INPUT_PATH, OUTPUT_PATH)

    print('Total time taken: {0:.3f} seconds'.format(time.time() - start_time))