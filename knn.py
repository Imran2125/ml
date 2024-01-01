import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()
print("\n Iris features target names\n",iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i, iris_dataset.target_names[i]))
print("\n Iris data:\n",iris_dataset["data"])
x_train,x_test,y_train,y_test=train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0)
print("Target:\n",iris_dataset["target"])
print("X_train\n",x_train)
print("X_test\n",x_test)
print("Y_train\n",y_train)
print("Y_test\n",y_test)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
for i in range(len(x_test)):
    x=x_test[i]
    x_new=np.array([x])
    prediction=knn.predict(x_new)
    print("\n Actual:{0} {1},predicted:{2} {3}".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][prediction]))
print("\n TestScore[Accuracy]:{:.2f}\n".format(knn.score(x_test, y_test)))                                                            