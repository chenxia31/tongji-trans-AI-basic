import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

plt.subplot(1,2,1)
data = np.load('./dataset/X.npy')
plt.scatter(data[:,0],data[:,1])
plt.subplot(1,2,2)
data = np.load('./dataset/Xval.npy')
label = np.load('./dataset/yval.npy')
plt.scatter(data[:,0],data[:,1],c=label.flatten(),cmap='rainbow')
plt.show()