#code
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
Y=df["y"].values
X=df.drop(columns=["y"]).values
x_train,x_test,y_train,y_test=train_test_split(X,Y)

x=pd.read_csv("test.csv").values
# print(X_test)
############# MODEL INSTANTEIATION ######################
KNN=KNeighborsClassifier(3)
KNN.fit(x_train,y_train)
accurecy=KNN.score(x_test,y_test)
prediction=pd.DataFrame(KNN.predict(x))
print(accurecy)



 