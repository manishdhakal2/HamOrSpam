from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


from Vectorize import Vectorizer
from Clean import Cleaner

#Load the data
data1=pd.read_csv(r"../data/Cleaned.csv")

#Separate into x and y
data_x=list(data1.iloc[:,1])
data_y=np.array(data1.iloc[:,0])

#Clean the data
cleanedData=Cleaner(data1)
#Join the cleaned Data
joinedCleanedData=[" ".join(i) for i in Cleaner(data1)]

#Vectorize the data
vector_x=Vectorizer(joinedCleanedData)

#Split the data

train_x,test_x,train_y,test_y=train_test_split(vector_x,data_y,test_size=0.33,random_state=42)







model=LogisticRegression()
model.fit(train_x,train_y)

y_pred=model.predict(test_x)

print(accuracy_score(test_y,y_pred))