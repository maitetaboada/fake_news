from gensim.models import KeyedVectors


def convertEncoding(embeddingFile, fromBinToTxt = True):
    #embeddingFile should be the name of your file excluding the format suffix

    if(fromBinToTxt):
        mw = KeyedVectors.load_word2vec_format(embeddingFile + ".bin", binary=True)
        mw.save_word2vec_format(embeddingFile + ".txt", binary=False)
    else:
        mw = KeyedVectors.load_word2vec_format(embeddingFile + ".txt", binary=False)
        mw.save_word2vec_format(embeddingFile + ".bin", binary=True)
    print "Conversion finished!"

def convertModel(inFile, outFile, fromW2VToGlove = True):
    #both files are txt
    if (fromW2VToGlove):
        fin = open(inFile, 'r')
        data = fin.read().splitlines(True)
        fout = open(outFile, 'w')
        fout.writelines(data[1:])
    else:
        print "This conversion not implemented yet!"
    print "Conversion finished!"                                                                    

convertEncoding("../pretrained/GoogleNews-vectors-negative300")
convertModel("../pretrained/GoogleNews-vectors-negative300.txt", "../pretrained/Gloved-GoogleNews-vectors-negative300.txt")


