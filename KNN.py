import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


Dataset=pd.DataFrame(pd.read_csv('student_performance.csv'))
Dataset.dropna(inplace=True)
Dataset['gender']=Dataset['gender'].replace({'Male':0,'Female':1})
Dataset['parent_education']=Dataset['parent_education'].replace({'High School':0,'Bachelor':1,'Master':2,'PhD':3})
Dataset['passed']=Dataset['passed'].replace({'No':0,'Yes':1})
# split the dataset into dependent and independent variables 
student_id=Dataset['student_id']
print(student_id)
X=Dataset.iloc[:,1:9]
Y=Dataset.iloc[:,10:11]

model = KNeighborsClassifier(3)
model.fit()









