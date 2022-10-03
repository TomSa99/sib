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
        return Dataset(data, label=False)

def write_data_file (dataset, filename, sep =',', label=False):
    '''Writes a Dataset object into a .txt file'''
    if label:
        x = dataset.x
        y = dataset.y
        data = np.concatenate((x, y), axis=1)
        savetxt(filename, data, delimiter=sep)
    else:
        savetxt(filename, dataset.x, delimiter=sep)

# test
if __name__ == '__main__':
    dataset = read_data_file('data.csv', sep =',', label=True)
    write_data_file(dataset, 'data.txt', sep =',', label=True)