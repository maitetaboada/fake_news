
from fastai import *
from fastai.text import * 
from textutils import DataLoading



path = untar_data(URLs.IMDB_SAMPLE)


# Preparing data using our own DataLoading functions


#*** fact-checking data ****
CLASSES = 2


texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_data_rubin()#load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_rubin()#load_data_liar("../data/liar_dataset/test.tsv")
texts_emergent, labels_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
#texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()
texts_train_snopes, labels_train_snopes = DataLoading.load_data_snopes("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES )#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/rashkin/xtrain.txt")#load_data_liar("../data/liar_dataset/train.tsv")#
texts_train_buzzfeed, labels_train_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
texts_train_emergent, labels_train_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
len(texts_train_snopes)
len(texts_train_buzzfeed)
len(texts_train_emergent)
print("Loading data is finished!")

print("Preparing training data...")
#texts_train = (pd.concat([pd.Series(texts_train_snopes) ,  pd.Series(texts_train_buzzfeed), pd.Series(texts_train_emergent)]))
#labels_train = (pd.concat([pd.Series(labels_train_snopes) ,  pd.Series(labels_train_buzzfeed), pd.Series(labels_train_emergent )]))
texts_snopes = (pd.concat([pd.Series(texts_train_snopes) ,   pd.Series(texts_snopesChecked),  pd.Series(texts_train_buzzfeed), pd.Series(texts_train_emergent)]))
labels_snopes = (pd.concat([pd.Series(labels_train_snopes) ,   pd.Series(labels_snopesChecked) ,  pd.Series(labels_train_buzzfeed), pd.Series(labels_train_emergent )]))
#***************************

# Language model data *********
#texts, labels = DataLoading.load_data_imdb()
texts_val, labels_val, texts_train, labels_train = DataLoading.balance_data(texts_snopes, labels_snopes, 50, [2,3,4,5])
train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
valid_df = pd.DataFrame( { 'label':  labels_val.astype(str), 'text':  texts_val})
print(train_df.head(100))
#data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_lm = TextLMDataBunch.from_df(path, train_df = train_df, valid_df = valid_df)

# Classifier model data ********
texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_snopes, labels_snopes, 400, [2,3,4,5])
texts_val, labels_val, texts, labels = DataLoading.balance_data(texts, labels, 100, [2,3,4,5])
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts, labels, 100,[2,3,4,5])
train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
valid_df = pd.DataFrame( { 'label':  labels_val.astype(str), 'text':  texts_val})
test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})
print("\n\n A sample of classifier training data:")
print(train_df.head(100))
#data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
data_clas = TextClasDataBunch.from_df(path, train_df = train_df, valid_df = valid_df, test_df = test_df, vocab=data_lm.train_ds.vocab, bs=32)

#print("Saving data to " + str(path) )
data_lm.save()
data_clas.save()




#print("Reading data from " + str(path) )

#data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=32)


#Building a language model
print("Language model learning")
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)

#Fit the model and save or load a trained model from disk
learn.fit_one_cycle(1, 1e-2)
print("Unfreez and fit one cycle")
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('ft_enc')



#Building a classifier
print("Building the text classifier")
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('ft_enc')

#Fit the model and save or load a trained model from disk
learn.fit_one_cycle(1, 1e-2)
learn.fit_one_cycle(1, 1e-3)
learn.fit_one_cycle(1, 1e-3/2.)

print("First freeze and fit cycle")
#learn.freeze_to(-2)
#learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
#learn.unfreeze()
#learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.fit(5, slice(2e-3/100, 2e-3))
learn.save('ft_classifier')

learn.load('ft_classifier')






#Predictions
sentence = "Hilary Clinton won the 2016 US election"
p = learn.predict(sentence)
print("Prediction for sentence \" " + sentence + "\" is " + str(p))


from sklearn.metrics import classification_report
def predict(mytexts):
    return [learn.predict(x)[0] for x in mytexts]

print("Results on training data:")
y = predict(texts_train)
print(classification_report( list(map(int, y)), labels_train))

print("Results on validation data:")
y = predict(texts_val)
print(classification_report( list(map(int, y)), labels_val))

print("Results on test data:")
y = predict(texts_test)
print(classification_report( list(map(int, y)), labels_test))

print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked , 40, [2,5])
y = predict(texts_test)
print(classification_report( list(map(int, y)), labels_test))


print("Test results on data sampled from emergent dataset (a broad distribution acc. to topic modeling -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_emergent, labels_emergent , 300, [2,5])
y = predict(texts_test)
print(classification_report( list(map(int, y)), labels_test))



print("Test results on data sampled from buzzfeed dataset (a narrow distribution : US election topic -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_train_buzzfeed, labels_train_buzzfeed , 70, [2,5])
y = predict(texts_test)
print(classification_report( list(map(int, y)), labels_test))


'''


import spacy
import torchtext 

spacy_tok = spacy.load('en')


m=learn.model
test_str ="bitcoin is bad"
tokenized_str = spacy_tok(test_str)
token_lst = [sent.string.strip() for sent in tokenized_str]

t = TEXT.numericalize([token_lst])
print(t)

# Set batch size to 1
m[0].bs=1

# Turn off dropout
m.eval()
# Reset hidden state
m.reset()
# Get predictions from model
res,*_ = m(t)
num_res = to_np(torch.topk(res[-1], 1)[1])[0]

print(num_res)





#learn.load('clas_2')
m = learn.model

#set batch size to 1
m[0].bs=1
#turn off dropout
m.eval()
#reset hidden state
m.reset()  

#sentence to be classified
sen = "Clinton won the 2016 US election"
idxs = np.array([[stoi[p] for p in sen.strip().split(" ")]])

#converting sentence into numerical representation
print('Numeric representation: ' + str(idxs))
idxs = np.transpose(idxs)

print('Value of idxs: ' + str(idxs)) 

#get predictions from model
p = m(VV(idxs))
print('Prediction: ' + str(p))
print(sen + ' - classification: ' + str(to_np(torch.topk(p[0],1)[1])[0]))

prob = F.softmax(p[0])
print('Probability rating for sentence: ' + str(prob))
'''
