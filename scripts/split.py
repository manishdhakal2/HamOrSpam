from sklearn.model_selection import train_test_split

def getSplitter(data):
    return train_test_split(data,test_size=0.33,random_state=42)


