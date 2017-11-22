
import sys
import os
import io

#sys.path = ['../utils'] + sys.path


import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize





class Word2vecFeatures:
    """
    The distributional bag of word model of text meaning:
    Vector representation of a text is obtained by adding up
    the vectors of its constituting words.
    """

    w2vModel = None

    def __init__(self, file):
        """

        :param file: pre-trained vectors file
        """

        print("Word2VecFeatures init: loading word2vec model")
        self.w2vModel = Word2Vec.load_word2vec_format(file, binary=True)  #gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=True)
        return

    def encode(self, texts):

        textvecs = list()
        textvec = None
        stoplist = stopwords.words('english')
        for index, text in enumerate(texts):
            #print(text)
            #print(type(text))
            text = text.lower().split()
            text = [i for i in text if i not in stoplist]
            wordCount = 0
            for word in text:
                if word in self.w2vModel.vocab:
                    if wordCount == 0:
                        textvec = self.w2vModel[word]
                    else:
                        textvec = np.add(textvec, self.w2vModel[word])
                    wordCount+=1
            if(wordCount == 0):
                #print(str(text))
                raise ValueError("Cannot encode text " + str(index) + " : all words unknown to model!  ::" + str(text))
            else:
                textvecs.append(normalize(textvec[:,np.newaxis], axis=0).ravel())
        return np.array(textvecs)
   
    def get_features(self, text):
        """
        :param text:
        :return: the exact avg vector of words in a text (later try other representations,e.g., mi)
        """

        features = self.encode([text])
        #print len(features[0]), " Word2vec features generated "
        return features[0]

     



"""
if __name__ == '__main__':

    m = Word2vecFeatures("/Users/fa/workspace/shared/sfu/fake_news/pretrained/embeddings/vectorsW.bin")
    print m.get_features("Greatings my dear lady!")
    print len(m.get_features("Book newspaper hello president cat health !"))


    
"""




    
    

