from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
data=pd.read_csv("ex.csv")
print("Input data and shape \n",data.shape)
print(data.head())
f1=data['V1'].values
f2=data['V2'].values
x=np.array(list(zip(f1,f2)))
print("x\n",x)
print("Graph for whole dataset\n")
plt.scatter(f1,f2,c='black',s=600)
plt.show()
kmeans=KMeans(2,random_state=0)
labels=kmeans.fit(x).predict(x)
print("Labels\n",labels)
centroids=kmeans.cluster_centers_
print("Centroids:\n",centroids)
plt.scatter(x[:,0],x[:,1], c=labels,s=40)
print("Graph using kmeans")
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='#050505')
plt.show()