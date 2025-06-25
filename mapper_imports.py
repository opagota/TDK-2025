import requests
import xarray as xr
import csv
import zipfile
import io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from PIL import Image
import numpy.ma as ma
from tqdm import tqdm
from itertools import product