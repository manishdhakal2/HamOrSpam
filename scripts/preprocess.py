from Clean import Cleaner
from split import getSplitter
import pandas as pd
import numpy as np

from Vectorize import Vectorizer

src="../Cleaned.csv"

data=pd.read_csv(src)

data[0]

train_X, test_X, train_y, test_y=getSplitter(data)

trainData=pd.DataFrame((train_X,train_y))
testData=pd.DataFrame((test_X,test_y))

trainData.to_csv("train.csv",index=False)

