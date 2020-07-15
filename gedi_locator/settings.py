import os
from dotenv import load_dotenv


load_dotenv()

PARTITIONS_05M_PATH = os.getenv('PARTITIONS_05M_PATH')
PARTITIONS_10M_PATH = os.getenv('PARTITIONS_10M_PATH')
PARTITIONS_20M_PATH = os.getenv('PARTITIONS_20M_PATH')
PARTITIONS_30M_PATH = os.getenv('PARTITIONS_30M_PATH')

COORDS_PATH = os.getenv('COORDS_PATH')

STORAGE_PATH = os.getenv('STORAGE_PATH')