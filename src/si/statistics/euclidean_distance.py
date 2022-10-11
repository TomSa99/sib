import numpy as np
# euclidean distance
def euclidean_distance(x, y):
    '''
Calcula a distância euclidiana entre X e Y usando a seguinte formula:
distance_y1n = np.sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
distance_y2n = np. sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
'''
    return np.sqrt((x - y)**2).sum(axis=1) 
    # axis=1 faz a soma das linhas, axis=0 faz a soma das colunas