import pickle
import spacy

nlp=spacy.load("en_core_web_sm")

def Cleaner(data):
    """
    Expects a Python List of Sentences
    """

    nlp=spacy.load("en_core_web_sm")

    

    cleanedData=[]

    for i in data:

        doc=nlp(i)

        text=[token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

        cleanedData.append(text)
    
    return cleanedData






