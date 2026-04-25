import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

Dataset=pd.DataFrame(pd.read_csv('student_performance.csv'))
Dataset.dropna(inplace=True)
Dataset['gender']=Dataset['gender'].map({'Male':0,'Female':1})
Dataset['parent_education']=Dataset['parent_education'].map({'High School':0,'Bachelor':1,'Master':2,'PhD':3})
Dataset['passed']=Dataset['passed'].map({'No':0,'Yes':1})
Dataset['internet_access']=Dataset['internet_access'].map({'No':0,'Yes':1})
Dataset['extracurricular']=Dataset['extracurricular'].map({'No':0,'Yes':1})
# split the dataset into dependent and independent variables 
student_id=Dataset['student_id']
print(student_id)
X=Dataset.iloc[:,1:9]
Y=Dataset.iloc[:,10].squeeze()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=15,stratify=Y)
train_accurecies={}
test_accurecies={}
for neighbors in np.arange(1,26):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train,y_train)
    train_accurecy=model.score(x_train,y_train)
    test_accurecy=model.score(x_test,y_test)
    train_accurecies[neighbors]=train_accurecy
    test_accurecies[neighbors]=test_accurecy


highest_train_accurecy=max(train_accurecies,key=train_accurecies.get)
print(highest_train_accurecy)
    
    












