from sib.src.si.data.dataset import Dataset
from scipy.stats import f_oneway

def f_classificiation (dataset: Dataset):
    classes = dataset.get_classes()
    # devolve as classes do dataset (valores possíveis de y)
    for samples in classes:
        f_oneway(samples)

