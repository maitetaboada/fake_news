
import sys
import os
import io

#sys.path = ['../utils'] + sys.path


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer





    
class TfIdfFeatures():
    """
    The tf/idf representation of text:
    TF/IDF vector for a text is obtained based on the entire reference corpus (collection of documents) given at initialization.
    This corpus can be the entire training corpus.
    """
    model = None

    def __init__(self, corpus):
        """

        :param corpus: list of documents
        """
        print("TfIdfFeatures() init: using the corpus...")
        print("Number of documents in the tf/idf corpus: " + str(len(corpus)))
        self.model = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1, stop_words='english')
        self.model.fit_transform(corpus)
        feature_names = self.model.get_feature_names()
        print("Number of terms used in the tf/idf table: " + str(len(feature_names)) )
        print feature_names[100:120]

    def encode(self, texts):
        print("transforming:...")
        #textvecs = []
        textvec = self.model.transform(texts)
        print(textvec)
            #textvecs.append(textvec)
        #print(textvecs)
        #return np.array(textvecs)

    def get_features(self, text):
        # obtains the bow distributional representation of the text
        # currently returns the exact avg vector of words in a text (later try other representations,e.g., mi)
        f = self.encode([text])
        print(f[0]) #" word2vec features generated "
        return f[0]
        
        
        
     



"""
if __name__ == '__main__':


    for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file = os.path.join(directory, filename)
                file_text = io.open(file,encoding = "ISO-8859-1").read()
                #print(file)
                corpus.append(file_text)
                continue
            else:
                continue
   

    xl = pd.ExcelFile("/Users/fa/workspace/temp/rubin/data.xlsx")
    df = xl.parse("Sheet1")
    corpus = df["Article Headline"]
    m = TfIdfFeatures(corpus)
    m.encode(corpus[1])
    m.encode("doctor hospital bad good newspaper!")
    m.encode(["doctor hospital bad good newspaper president health !"])
    print m.get_features("doctor hospital bad good newspaper!")
    print len(m.get_features("doctor hospital bad good newspaper!"))
    
"""




    
    

