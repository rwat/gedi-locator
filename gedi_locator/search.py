import json
import numpy as np
from numba import jit, prange, int8, float64, int64, void
import os
import random
from time import sleep
from timeit import default_timer as timer

import settings
from shared import bbox_intersect, get_filenames, is_overlap_sorted_values, poly_bbox


INPUT_PATH = settings.PARTITIONS_05M_PATH
STORAGE_PATH = settings.STORAGE_PATH

NUM_STORAGE_FILES = 4

# partitions: [('granules_idx'), ('polynomial', (8)), ('bbox', (4)), ('max_error')] == 14 wide
PARTITIONS_WIDTH = 14
GRANULES_IDX = 0
POLYNOMIAL_BEGIN = 1
POLYNOMIAL_END = 9
P_BBOX_BEGIN = 9
P_BBOX_END = 13
MAX_ERROR = 13
# granules:   [('bbox', (4)), ('partitions_offset_left'), ('partitions_offset_right')] == 6 wide
GRANULES_WIDTH = 6
G_BBOX_BEGIN = 0
G_BBOX_END = 4
PARTITIONS_OFFSET_LEFT = 4
PARTITIONS_OFFSET_RIGHT = 5


def _orbit_bbox(partitions):
    """ Takes a granule's partitions 'partitions' and returns the bounding box
        containing all of them. Bounding box is ll, ur format
        [[lon, lat], [lon, lat]]. """
    lon_min = partitions[0]['lon_min']
    lat_min = partitions[0]['lat_min']
    lon_max = partitions[0]['lon_max']
    lat_max = partitions[0]['lat_max']
    for p in partitions[1:]:
        if p['lon_min'] < lon_min:
            lon_min = p['lon_min']
        if p['lat_min'] < lat_min:
            lat_min = p['lat_min']
        if p['lon_max'] > lon_max:
            lon_max = p['lon_max']
        if p['lat_max'] > lat_max:
            lat_max = p['lat_max']
    return [[lon_min, lat_min], [lon_max, lat_max]]


def write_data(urls, granules, partitions, partitions_lons_max, min_lat, max_lat):
    # write files to disk in order speed up loading on subsequent executions
    np.save(os.path.join(STORAGE_PATH, 'granules.npy'), granules)
    np.save(os.path.join(STORAGE_PATH, 'partitions.npy'), partitions)
    np.save(os.path.join(STORAGE_PATH, 'partitions_lons_max.npy'), partitions_lons_max)
    with open(os.path.join(STORAGE_PATH, 'obj.json'), 'w') as f:
        obj = {'urls': urls,
               'min_lat': min_lat,
               'max_lat': max_lat}
        json.dump(obj, f)


def read_data():
    granules = np.load(os.path.join(STORAGE_PATH, 'granules.npy'))
    partitions = np.load(os.path.join(STORAGE_PATH, 'partitions.npy'))
    partitions_lons_max = np.load(os.path.join(STORAGE_PATH, 'partitions_lons_max.npy'))
    with open(os.path.join(STORAGE_PATH, 'obj.json')) as f:
        obj = json.load(f)
    return obj['urls'], granules, partitions, partitions_lons_max, \
            obj['min_lat'], obj['max_lat']


def load_data(input_path):
    """ Takes a filesystem path input_path and returns a tuple of
        (urls, granules, partitions, partitions_lons_max, min_lat,
        max_lat). """

    if len(os.listdir(STORAGE_PATH)) == NUM_STORAGE_FILES:
        return read_data()

    # NOTE: simulating two years of data with 1 month repeated
    dataset_multiplier = 24

    # Determine the dimensions of ndarrays
    filenames = get_filenames(input_path)
    granules_count = 0
    partitions_count = 0
    for f in filenames:
        granules_count += 1
        with open(input_path + f) as g:
            partitions_count += len(json.load(g))
    
    granules_count = granules_count * dataset_multiplier
    partitions_count = partitions_count * dataset_multiplier
    
    # Create ndarrays
    partitions = np.zeros((partitions_count, PARTITIONS_WIDTH))
    granules = np.zeros((granules_count, GRANULES_WIDTH))

    # Populate data
    urls = []
    partitions_lons_max = np.zeros((partitions_count))
    granules_idx = 0
    partitions_idx = 0
    for _ in range(dataset_multiplier):
        for f in filenames:
            # append to list of urls
            urls.append('https://placeholder.url/' + f)
            # assign values to granule and partitions
            with open(input_path + f) as g:
                partitions_json = json.load(g)
            granules[granules_idx][PARTITIONS_OFFSET_LEFT] = partitions_idx
            granules[granules_idx][PARTITIONS_OFFSET_RIGHT] = partitions_idx + len(partitions_json) - 1
            for p in partitions_json:
                # set all the values in this partition
                partitions_view = partitions[partitions_idx]
                partitions_view[GRANULES_IDX] = granules_idx
                np.put(partitions_view, np.arange(POLYNOMIAL_BEGIN, POLYNOMIAL_END), tuple(p['pn']))
                np.put(partitions_view, np.arange(P_BBOX_BEGIN, P_BBOX_END), [p['lon_min'], p['lat_min'], p['lon_max'], p['lat_max']])
                partitions_view[MAX_ERROR] = p['max_error']
                # set a separate value for binary search
                partitions_lons_max[partitions_idx] = p['lon_max']
                # increment for the next partition
                partitions_idx += 1
            np.put(granules[granules_idx], np.arange(G_BBOX_BEGIN, G_BBOX_END), np.array(_orbit_bbox(partitions_json)).flatten())
            granules_idx += 1

    # Find the minimum latitude and maximum latitude across all partitions
    min_lat = 0
    max_lat = 0
    for i in range(0, len(granules)):
        minl = (granules[i][G_BBOX_BEGIN:G_BBOX_END])[1]
        maxl = (granules[i][G_BBOX_BEGIN:G_BBOX_END])[3]
        if min_lat > minl:
            min_lat = minl
        if max_lat < maxl:
            max_lat = maxl
    
    # write files to disk in order speed up loading on subsequent executions
    write_data(urls, granules, partitions, partitions_lons_max, min_lat, max_lat)

    return urls, granules, partitions, partitions_lons_max, min_lat, max_lat


@jit
def _search(granules_idx, aoi_bbox, granules, partitions, partitions_lons_max, matched_granules):
    aoi_lon_min = aoi_bbox[0][0]
    aoi_lon_max = aoi_bbox[1][0]
    aoi_lat_min = aoi_bbox[0][1]
    aoi_lat_max = aoi_bbox[1][1]

    g_bbox = (granules[granules_idx][G_BBOX_BEGIN:G_BBOX_END]).copy().reshape((2, 2))
    
    # Does aoi_bbox intersect this granule's bbox
    if bbox_intersect(aoi_bbox, g_bbox):
        left_idx = int(round(granules[granules_idx][PARTITIONS_OFFSET_LEFT]))
        right_idx = int(round(granules[granules_idx][PARTITIONS_OFFSET_RIGHT] + 1))
        partitions_view = partitions[left_idx:right_idx]
        partitions_lons_max_view = partitions_lons_max[left_idx:right_idx]
        # Binary search on sorted values
        start_idx = np.searchsorted(partitions_lons_max_view, aoi_lon_min)
        # Iterate through relevant partitions
        for i in range(start_idx, right_idx+1):
            # Check for longitude overlap. NOTE: is_overlap_sorted_values is
            #   not used here because NaN raw data values can cause partition
            #   splits resulting in a premature end to this loop if we were
            #   to use that function.
            p_lon_min = (partitions_view[i][P_BBOX_BEGIN:P_BBOX_END])[0]
            if aoi_lon_max > p_lon_min:
                # Check for whole partition bbox overlap with aoi_bbox
                p_bbox = (partitions_view[i][P_BBOX_BEGIN:P_BBOX_END]).copy().reshape((2, 2))
                if bbox_intersect(p_bbox, aoi_bbox):
                    # Check for specific calculated swath overlap. Generate a
                    # bounding box for the polynomial that is:
                    #  - the greater of aoi_lon_min, p_lon_min
                    #  - the lesser of aoi_lon_max, p_lon_max
                    # p_g_ is short for polynomial, generated
                    p_lon_max = (partitions_view[i][P_BBOX_BEGIN:P_BBOX_END])[2]
                    p_g_lon_min = aoi_lon_min if aoi_lon_min > p_lon_min else p_lon_min
                    p_g_lon_max = aoi_lon_max if aoi_lon_max < p_lon_max else p_lon_max
                    p_g_bbox = poly_bbox(partitions_view[i][POLYNOMIAL_BEGIN:POLYNOMIAL_END], p_g_lon_min, p_g_lon_max, partitions_view[i][MAX_ERROR])

                    # We know specific longitudes of each bbox overlap; now
                    #   detect specific latitude overlap.
                    p_g_lat_min = p_g_bbox[0][1]
                    p_g_lat_max = p_g_bbox[1][1]
                    if is_overlap_sorted_values(p_g_lat_min, p_g_lat_max, aoi_lat_min, aoi_lat_max):
                        # aoi_bbox and partition overlap; update matched_granules and end search for this granule
                        matched_granules[granules_idx] = True
                        break
            else:
                break


@jit(nopython=True, nogil=True, parallel=True)
def search(aoi_bbox, granules, partitions, partitions_lons_max, granules_bbox):
    # An array that will have each element set to 1 according to each granule
    #   that matches a given aoi_bbox.
    matched_granules = np.zeros(len(granules), dtype=np.bool_)

    # Verify the aoi_bbox is within at least one of the granules_bbox
    if bbox_intersect(aoi_bbox, granules_bbox):
        # parallel processing across granules
        for granules_idx in prange(len(granules)):
            _search(granules_idx, aoi_bbox, granules, partitions, partitions_lons_max, matched_granules)

    return np.nonzero(matched_granules)[0]


def granules_to_urls(granules_idxs, urls):
    """ Takes an array of granules indexes granules_idx and urls array 'urls'
        and returns a list of urls that correspond to the indices. """
    results = []
    for idx in granules_idxs:
        results.append(urls[idx])
    return results


def random_bbox():
    """ Returns random bbox around part of the amazon (roughly) in ll, ur
        format [[lon, lat], [lon, lat]]. """
    lons_fn = lambda: random.uniform(-74.0, -70)
    lats_fn = lambda: random.uniform(1.1, 3.9)
    lons = sorted((lons_fn(), lons_fn()))
    lats = sorted((lats_fn(), lats_fn()))
    return np.array([[lons[0], lats[0]], [lons[1], lats[1]]])


def benchmark():
    test_bboxes = [np.array([[-74.1, 3.1], [-73.7, 3.3]]),
                   np.array([[-74.1, 2.9], [-73.3, 3.3]]),
                   np.array([[-74.1, 2.7], [-72.9, 3.3]]),
                   np.array([[-74.1, 2.5], [-72.5, 3.3]]),
                   np.array([[-74.1, 2.3], [-72.1, 3.3]]),
                   np.array([[-74.1, 2.1], [-71.7, 3.3]]),
                   np.array([[-74.1, 1.9], [-71.3, 3.3]]),
                   np.array([[-74.1, 1.7], [-70.9, 3.3]]),
                   np.array([[-74.1, 1.5], [-70.5, 3.3]]),
                   np.array([[-74.1, 1.3], [-70.1, 3.3]])]
    
    rand_bboxes = [random_bbox() for _ in range(10)]

    # Load data
    start_time = timer()
    urls, granules, partitions, partitions_lons_max, min_lat, max_lat = load_data(INPUT_PATH)
    end_time = timer()
    print('Load time: {0:.3f} seconds'.format(end_time - start_time))

    #sleep(3)

    granules_bbox = np.array([[-180.0, min_lat], [180.0, max_lat]])

    # warm up for Numba and JIT
    for _ in range(10):
        for t in test_bboxes:
            granules_idxs = search(t, granules, partitions, partitions_lons_max, granules_bbox)

    # establish search time    
    start_time = timer()
    for b in rand_bboxes:
        granules_idxs = search(b, granules, partitions, partitions_lons_max, granules_bbox)
        granules_to_urls(granules_idxs, urls)
    end_time = timer()
    print('Search time: {0:.9f} seconds'.format((end_time - start_time) / len(test_bboxes)))

    # return URLs for a search
    granules_idxs = search(test_bboxes[0], granules, partitions, partitions_lons_max, granules_bbox)
    print(sorted(granules_to_urls(granules_idxs, urls)))


if __name__ == "__main__":
    benchmark()
