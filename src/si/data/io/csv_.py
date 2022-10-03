import pandas as pd
import numpy as np
from si.data.dataset import Dataset

# pandas.read_csv(https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

# pd.read_csv('/Users/Utilizador/Desktop/SIB/notas.csv', sep=',')

def read_csv(filename, sep = ',', features=False, label=False ):
    '''read csv file and return a dataframe'''
    if features and label:
        df=pd.read_csv(filename, sep=sep)
        features = df.columns[:-1]
        labels = df.columns[-1]
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return Dataset(x, y, features, labels)
    elif features:
        df=pd.read_csv(filename, sep=sep)
        features = df.columns [:].values
        x = df.iloc[:, :].values
        return Dataset(x, features=features)
    elif label:
        df=pd.read_csv(filename, sep=sep)
        labels = df.columns[-1]
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return Dataset(y, labels=labels)
    else:
        df=pd.read_csv(filename, sep=sep)
        x = df.iloc[:, :].values
        return Dataset(df)
    
def write_csv(filename, dataset, sep = ',', features = False, label = False):
    '''write csv file'''
    if features and label:
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df[dataset.labels] = dataset.y
        df.to_csv(filename, sep=sep, index=False)
    elif features:
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df.to_csv(filename, sep=sep, index=False)
    elif label:
        df = pd.DataFrame(dataset.y, columns=dataset.labels)
        df.to_csv(filename, sep=sep, index=False)
    else:
        df = pd.DataFrame(dataset.x)
        df.to_csv(filename, sep=sep, index=False)
    
"testar o código"
if __name__ == '__main__':
    # read_csv
    dataset = read_csv('/Users/Utilizador/Desktop/SIB/notas.csv', sep=',', features=True, label=True)
    print(dataset.summary())

    # write_csv
    write_csv('/Users/Utilizador/Desktop/SIB/notas2.csv', dataset, sep=',', features=True, label=True)    

read_csv('/Users/Utilizador/Desktop/SIB/notas.csv')