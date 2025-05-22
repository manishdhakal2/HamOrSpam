from sklearn.feature_extraction.text import TfidfVectorizer

model=TfidfVectorizer()

def Vectorizer(data:list):
    """
    Expects a python list of lemmatized texts
    """

    vectorized=model.fit_transform(data)


    return vectorized

    