import pandas as pd
import numpy as np

Dataset=pd.DataFrame(pd.read_csv('student_performance.csv'))
Dataset.dropna(inplace=True)

# split the dataset into dependent and independent variables 
student_id=Dataset['student_id']
print(student_id)
X=Dataset.iloc[:,1:9]
Y=Dataset.iloc[:,9:11]
print(X)



