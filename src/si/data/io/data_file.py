import pandas as pd
import numpy as np
from numpy import genfromtxt
from si.data.dataset import Dataset
from numpy import savetxt

def read_data_file (filename, sep =',',label=False):
    '''Reads a generic file, transforms it into a .txt file and returns a Dataset object'''
    if label:
        data = genfromtxt(filename, delimiter=sep)
        x = data[:, :-1]
        y = data[:, -1]
        return Dataset(x, y, label=True)
    else:
        data = genfromtxt(filename, delimiter=sep)
        return Dataset(data)