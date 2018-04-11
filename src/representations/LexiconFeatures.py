
import sys
import os
import io

#sys.path = ['../utils'] + sys.path


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import nltk





    
class LexiconFeatures():
    """
    The tf/idf representation of text:
    TF/IDF vector for a text is obtained based on the entire reference corpus (collection of documents) given at initialization.
    This corpus can be the entire training corpus.
    """
    lexicons = None

    def __init__(self, lexicon_directory):
        """

        :param corpus: list of documents
        """
        self.lexicons = []
        print("LexiconFeatures() init: loading lexicons")
        for filename in os.listdir(lexicon_directory):
            if filename.endswith(".txt"):
                file = os.path.join(lexicon_directory, filename)
                lexicon = {k: 0 for k in nltk.word_tokenize(open(file).read())}
                print("Number of terms in the lexicon " + filename + " : " + str(len(lexicon)))
                self.lexicons.append(lexicon)
                continue
            else:
                continue



    def encode(self, texts):
        stoplist = stopwords.words('english')
        #print("transforming:...")
        textvecs = []
        for text in texts:
            #print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = []
            for lexicon in self.lexicons:
                lexicon_words = [i for i in tokens if i in lexicon]
                count = len(lexicon_words)
                #print ("Count: " + str(count))
                count = count * 1.0 / len(tokens)  #Continous treatment
                #count = 1 if (count > 0) else 0     #Binary treatment
                textvec.append(count)
            #print(textvec)
            textvecs.append(textvec)
        return np.array(textvecs)

    def get_features(self, text):
        # obtains the bow distributional representation of the text
        # currently returns the exact avg vector of words in a text (later try other representations,e.g., mi)
        f = self.encode([text])
        #print(f[0]) #" word2vec features generated "
        return f[0]
        
        
        
     



'''
if __name__ == '__main__':




    xl = pd.ExcelFile("/Users/fa/workspace/temp/rubin/data.xlsx")
    df = xl.parse("Sheet1")
    corpus = df["Full Text"]

    m = LexiconFeatures("/Users/fa/workspace/temp/NPOV/bias_related_lexicons")
    m.encode(corpus[1:7])
    m.encode(["doctor guess hospital bad good newspaper!"])
    m.encode(["doctor seem appear hospital bad good newspaper president health !"])
    print m.get_features("guess seem appear know one two!")
    #print len(m.get_features("doctor hospital bad good newspaper!"))
    
'''




    
    

