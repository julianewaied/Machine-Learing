import pandas as pd
import numpy as np
from matplotlib import pyplot as pt
# import a dataset of points
data_path = "C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Python Tests/training_set.csv"
data = pd.read_csv(data_path)
data = np.array(data)
data = data - np.mean(data)
mat = np.cov(data)
mat = (1/len(data)) * mat
vals,vecs = np.linalg.eigh(mat)
# so that v[0] is the minimal
vecs = vecs[::-1]
print(vecs[0]/vecs[0,0])