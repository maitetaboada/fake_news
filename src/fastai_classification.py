
from fastai import *
from fastai.text import * 
from textutils import DataLoading


#path = untar_data(URLs.IMDB_SAMPLE)
path =  "~/workspace/shared/sfu/fake_news/dump/fastai"

# Preparing data using our own DataLoading functions


#*** fact-checking data ****
CLASSES = 2
LOAD_DATA_FROM_DISK = True


if LOAD_DATA_FROM_DISK:
	texts_train = np.load("../dump/trainRaw")
	texts_valid = np.load("../dump/validRaw")
	texts_test = np.load("../dump/testRaw")
	labels_train = np.load("../dump/trainlRaw")
	labels_valid = np.load("../dump/validlRaw")
	labels_test = np.load("../dump/testlRaw")
	print("Data loaded from disk!")

else:
	# Data sources used for training:
	texts_train_snopes, labels_train_snopes = DataLoading.load_data_snopes("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES )#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/r$
	texts_train_buzzfeed, labels_train_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
	texts_train_emergent, labels_train_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
	len(texts_train_snopes)
	len(texts_train_buzzfeed)
	len(texts_train_emergent)
	
	texts_all_train = pd.concat([pd.Series(texts_train_snopes) ,  pd.Series(texts_train_buzzfeed), pd.Series(texts_train_emergent)])
	labels_all_train = pd.concat([pd.Series(labels_train_snopes) ,  pd.Series(labels_train_buzzfeed), pd.Series(labels_train_emergent )])
	#sources_all_train = pd.concat()

	texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_all_train, labels_all_train, 1200, [2,3,4,5])
	texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 400, [2,3,4,5])
	texts_test, labels_test, texts, labels = DataLoading.balance_data(texts, labels, 400,[2,3,4,5])
	texts_train.dump("../dump/trainRaw")
	texts_valid.dump("../dump/validRaw")
	texts_test.dump("../dump/testRaw")
	labels_train.dump("../dump/trainlRaw")
	labels_valid.dump("../dump/validlRaw")
	labels_test.dump("../dump/testlRaw")
	print("Data dumped to disk!")


#******************************
#****Language model************
data_lm = TextLMDataBunch.load(path + "/languageModel")
'''
# Use training data for language model tuning (or some external big unlabeled corpus)
#train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
#valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
#test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})
texts_external, labels_external = DataLoading.load_data_rashkin("../data/rashkin/xtrain.txt")
texts_external, labels_external, texts, labels = DataLoading.balance_data(texts_external, labels_external, 5000)
train_df = pd.DataFrame( {'label':  labels_external.astype(str), 'text':  texts_external})
valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})

print(train_df.head(10))
data_lm = TextLMDataBunch.from_df(path + "/languageModel", train_df = train_df, valid_df = valid_df, test_df = test_df)
data_lm.save()

# Building a language model
print("Language model learning")
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
# Fit the model and save on disk
learn.fit_one_cycle(1, 1e-2)
print("Unfreez and fit one cycle")
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('ft_enc')
'''

#******************************
#****Classifier model**********
print("Reading classification data from " + str(path) )
data_clas = TextClasDataBunch.load(path + "/classification", bs=32)

train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})
print("\n\n A sample of classifier training data:")
print(train_df.head(100))
data_clas = TextClasDataBunch.from_df(path + "/classification", train_df = train_df, valid_df = valid_df, test_df = test_df, vocab=data_lm.train_ds.vocab, bs=32)
print("Saving classification data to " + str(path) )
data_clas.save()


# Building a classifier
print("Building the text classifier")
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder("../../languageModel/models/"+ 'ft_enc')
# Fit the model and save or load a trained model from disk
learn.load('ft_classifier')
'''
learn.fit_one_cycle(1, 1e-2)
learn.fit_one_cycle(1, 1e-3)
learn.fit_one_cycle(1, 1e-3/2.)
print("First freeze and fit cycle")
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.fit(5, slice(2e-3/100, 2e-3))
#learn.save('ft_classifier')
'''
#learn.unfreeze()
learn.fit_one_cycle(1, 1e-2)
learn.fit_one_cycle(1, 1e-3)
learn.fit_one_cycle(1, 1e-3/2.)
learn.fit(10, slice(2e-3/100, 2e-3))
learn.save('ft_classifier')

#Predictions
sentence = "Hilary Clinton won the 2016 US election"
p = learn.predict(sentence)
print("Prediction for sentence \" " + sentence + "\" is " + str(p))


from sklearn.metrics import classification_report, confusion_matrix

def predict(mytexts):
    return [learn.predict(x)[0] for x in mytexts]

'''
print("Reading classification data from " + str(path) )
data = TextClasDataBunch.load(path + "/classification" , bs=32)
print("Results on training data:")
x,yhat = next(iter(data.train_dl))
y = learn.get_preds(ds_type= DatasetType.Train)[0]
print("output of get_preds")
print(y)
print(torch.max(y))
#y = predict(x)
print(classification_report(yhat,   torch.max(y)))
'''

print("Results on training data:")
y = predict(texts_train)
print(classification_report(labels_train,  list(map(int, y))))

print("Results on validation data:")
y = predict(texts_valid)
print(classification_report( labels_valid, list(map(int, y))))

print("Results on test data:")
y = predict(texts_test)
print(classification_report( labels_test, list(map(int, y))))


# Data sources used for testing:
texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_d$
texts_emergent, labels_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
texts_buzzfeed, labels_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
#texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()

print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked , 40, [2,5])
y = predict(texts_test)
print(classification_report( labels_test, list(map(int, y))))
texts_test, labels_test =  DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", classes = 5)
y = predict(texts_test)
print("confusion matrix:")
print(confusion_matrix(labels_test, list(map(int, y))))
print(pd.DataFrame({'Predicted': list(map(int, y)), 'Expected': labels_test}))

print("Test results on data sampled from emergent dataset (a broad distribution acc. to topic modeling -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_emergent, labels_emergent , 300, [2,5])
y = predict(texts_test)
print(classification_report( labels_test, list(map(int, y))))

print("Test results on data sampled from buzzfeed dataset (a narrow distribution : US election topic -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_buzzfeed, labels_buzzfeed , 70, [2,5])
y = predict(texts_test)
print(classification_report( labels_test, list(map(int, y))))


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
