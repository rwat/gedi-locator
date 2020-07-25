# GEDI geo-locator

GEDI is a remote sensing instrument on the ISS and this is one approach for a GEDI version 1 geo-locator prototype.

To learn more about the code in this repo you'll want to read the paper at [Element 84](https://element84.com/blog/gedi).

## Usage Instructions

Here are the steps for using this prototype software:

1. Download GEDI L1B granules
2. Extract Lat/Lon coordinates from each granule
3. Run polynomial_gen_json.py on the extracted coordinates json to create orbital partitions data
4. Run the search PoC which first loads the partitions data and then runs some benchmark searches

Alternatively and for testing purposes I've provided pre-generated JSON for 5m error dynamic partitions (see paper above to know what that means) for data from the month of May 2019. Download either the .tar.gz or .zip, extract to a directory of your choice and then run `search.py` with the extracted contents as the `INPUT_PATH`.

[gedi_json_dynpart_2019_05__05m.tar.gz](https://rwat.s3-us-west-2.amazonaws.com/github/gedi-locator/gedi_json_dynpart_2019_05__05m.tar.gz)

[gedi_json_dynpart_2019_05__05m.zip](https://rwat.s3-us-west-2.amazonaws.com/github/gedi-locator/gedi_json_dynpart_2019_05__05m.zip)

However, if you'd like to start from GEDI data please see the directions below.


### Download GEDI L1B granules

GEDI L1B granules can be downloaded for free from NASA's LP DAAC Data Pool:

[https://e4ftl01.cr.usgs.gov/GEDI/GEDI01_B.001/](https://e4ftl01.cr.usgs.gov/GEDI/GEDI01_B.001/)

In order to download granules you'll need a NASA Earthdata Login if you don't already have one - it's free and just requires you to provide an email address and password. Further instructions can be found on Data Pool.

One of the easiest ways to download granules is by using `wget` and a file with all the urls you'd like to download.

The following is some example python which can crawl Data Pool to extract L1B granules from the above link using the `requests` library:


```
def assemble_gedi_url_list():
    urls = []

    base_url = 'https://e4ftl01.cr.usgs.gov/GEDI/GEDI01_B.001/'
    dir_text = (requests.get(base_url)).text

    regex = re.compile('a href="([\d\.\/]+)"')
    dirs = [regex.search(line).group(1) for line in dir_text.splitlines() if regex.search(line)]

    for dir in dirs:
        regex = re.compile('a href="(.+?\.h5)"')
        dir_url = base_url + dir
        granule_text = (requests.get(dir_url)).text
        granules = [regex.search(line).group(1) for line in granule_text.splitlines() if regex.search(line)]

        urls = urls + [dir_url + granule + '\n' for granule in granules]
    
    with open('./gedi_urls.txt', 'w') as f:
        f.writelines(urls)
```

Once you have a list of URLs you'd like to download, the following `wget` command will download them:

```
wget -c -i ../gedi-locator/gedi_urls.txt --user=your.earthdata.username --ask-password
```

Just a heads up that each granule will be about 6GB to 10GB in size or larger.

### Extract Lat/Lon Coordinates From Each Granule

Once you've acquired granules you can use `h5py` to extract layer data and store them as json:

```
def extract_coords():
    from_path = '/your/from/path/here'
    to_path = '/your/to/path/here/gedi_l1b_coords'

    for root, dirs, files in os.walk(from_path):
        if len(files) > 0:
            for f in files:
                with open(os.path.join(root, f), 'rb') as g:
                    granule = h5py.File(g, 'r')
                    output_name = re.sub(r'\.h5$', '', f)
                    print(output_name)
                    with open(os.path.join(to_path, output_name), 'w') as output_file:
                        output_data = {'name': output_name,
                                       'lons': granule['BEAM0110/geolocation/longitude_bin0'][()].tolist(),
                                       'lats': granule['BEAM0110/geolocation/latitude_bin0'][()].tolist()}
                        json.dump(output_data, output_file)
```

### Fit Polynomials By Dynamically-Sized Partition

Take the Lat/Lon layers for each orbit and fit polynomials to partitioned orbits with `polynomial_get_json.py`. The way it's currently written it will run one process per orbit at a concurrency of `multiprocessing.cpu_count()`. In a `.env` file you'll need to specify your source and destination directories in accordance with `settings.py`.

### Run the Search PoC

You can now run `search.py` to load in the orbit partitions and geo-locate for an AOI bounding box. Again, look to your local `.env` file and `settings.py` so the search will look to the correct directories.


## TODO
- unit tests
- web service?
