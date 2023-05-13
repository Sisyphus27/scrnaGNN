import numpy as np
import pandas as pd
import os
from pandas.api.types import is_string_dtype, is_numeric_dtype
import networkx as nx
import seaborn as sns
import pylab as plt
import matplotlib as mpl
import shapely.geometry as geom
from copy import deepcopy
import itertools
from scipy.spatial import distance, cKDTree, KDTree
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, UnivariateSpline
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import coo_matrix, diags
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import math
from decimal import *


def get_extension(filename):
    fn, ext = os.path.splitext(filename)
    while (ext[1:] in ['gz', 'bz2', 'zip', 'xz']):
        fn, ext = os.path.splitext(fn)
    return ext[1:]
