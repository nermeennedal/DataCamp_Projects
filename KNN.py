#code
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
Y_train=df["y"].values
X_train=df.drop(columns=["y"]).values
X_test=pd.read_csv("test.csv").values
# print(X_test)
############# MODEL INSTANTEIATION ######################
KNN=KNeighborsClassifier(2)
KNN.fit(X_train,Y_train)
prediction=pd.DataFrame(KNN.predict(X_test))
print(prediction)
