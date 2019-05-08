
from fastai import *
from fastai.text import *
from textutils import DataLoading
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

#reprudicibility:

import random
import numpy as np
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)





#path = untar_data(URLs.IMDB_SAMPLE)
path =  "~/workspace/shared/sfu/fake_news/dump/fastai"

# Preparing data using our own DataLoading functions


#*** fact-checking data ****
CLASSES = 2
LOAD_DATA_FROM_DISK = True
LOAD_LM_FROM_DISK = True
LOAD_TC_FROM_DISK = False

my_drop_mult = 0.5
my_lr = 2e-3 # 1e-2,2e-3
my_epochs = 30


if LOAD_DATA_FROM_DISK:
	texts_train = np.load("../dump/trainRaw")
	texts_valid = np.load("../dump/validRaw")
	texts_test = np.load("../dump/testRaw")
	labels_train = np.load("../dump/trainlRaw")
	labels_valid = np.load("../dump/validlRaw")
	labels_test = np.load("../dump/testlRaw")
	texts_train_external = np.load("../dump/trainRaw_external")
	texts_valid_external = np.load("../dump/validRaw_external")
	texts_test_external = np.load("../dump/testRaw_external")
	labels_train_external = np.load("../dump/trainlRaw_external")
	labels_valid_external = np.load("../dump/validlRaw_external")
	labels_test_external = np.load("../dump/testlRaw_external")
	print("Data loaded from disk!")

else:
	# Data sources used for training:
	texts_train_snopes, labels_train_snopes = DataLoading.load_data_snopes("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES )
	texts_train_buzzfeed, labels_train_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
	texts_train_emergent, labels_train_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
	len(texts_train_snopes)
	len(texts_train_buzzfeed)
	len(texts_train_emergent)

	texts_all = pd.concat([pd.Series(texts_train_snopes) ,  pd.Series(texts_train_buzzfeed), pd.Series(texts_train_emergent)])
	labels_all = pd.concat([pd.Series(labels_train_snopes) ,  pd.Series(labels_train_buzzfeed), pd.Series(labels_train_emergent )])
	#sources_all_train = pd.concat()

	texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_all, labels_all, 1200, [2,3,4,5])
	texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 400, [2,3,4,5])
	texts_test, labels_test, texts, labels = DataLoading.balance_data(texts, labels, 400,[2,3,4,5])
	texts_train.dump("../dump/trainRaw")
	texts_valid.dump("../dump/validRaw")
	texts_test.dump("../dump/testRaw")
	labels_train.dump("../dump/trainlRaw")
	labels_valid.dump("../dump/validlRaw")
	labels_test.dump("../dump/testlRaw")

	'''
	texts_all_external, labels_all_external = DataLoading.load_data_rashkin("../data/rashkin/train.txt", CLASSES)
	
	texts_train_external, labels_train_external, texts, labels = DataLoading.balance_data(texts_all_external, labels_all_external, 6000)
	texts_valid_external, labels_valid_external, texts, labels = DataLoading.balance_data(texts, labels, 400)
	texts_test_external, labels_test_external, texts, labels = DataLoading.balance_data(texts, labels, 400)
	texts_train_external.dump("../dump/trainRaw_external")
	texts_valid_external.dump("../dump/validRaw_external")
	texts_test_external.dump("../dump/testRaw_external")
	labels_train_external.dump("../dump/trainlRaw_external")
	labels_valid_external.dump("../dump/validlRaw_external")
	labels_test_external.dump("../dump/testlRaw_external")
	'''
	print("Data dumped to disk!")


#******************************
#****Language model************
if (LOAD_LM_FROM_DISK):
	print("Reading language model data from " + str(path) )
	data_lm = TextLMDataBunch.load(path + "/languageModel")
else:
	# Use training data for language model tuning (or some external big [unlabeled] corpus)
	#train_df = pd.DataFrame( {'label':  labels_train_external.astype(str), 'text':  texts_train_external})
	#valid_df = pd.DataFrame( { 'label':  labels_valid_external.astype(str), 'text':  texts_valid_external})
	#test_df = pd.DataFrame( {'label':  labels_test_external.astype(str), 'text':  texts_test_external})
	train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
	valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
	test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})
	
	print(train_df.head(10))
	data_lm = TextLMDataBunch.from_df(path + "/languageModel", train_df = train_df, valid_df = valid_df, test_df = test_df)
	data_lm.save()

	# Building a language model
	print("Language model learning")
	learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
	# Fit the model and save on disk
	learn.freeze_to(-1)
	learn.fit_one_cycle(1, 1e-3/2)
	learn.unfreeze()
	learn.fit(10, 1e-3)
	learn.save_encoder('LM_selfData')

#******************************
#****Classifier model**********

texts_train_valid = pd.concat([pd.Series(texts_train), pd.Series(texts_valid)])
labels_train_valid = pd.concat([pd.Series(labels_train), pd.Series(labels_valid)])

print(texts_train_valid[0:3][0:10])
print(labels_train_valid[0:3])

def predict(learn, mytexts):
    return [learn.predict(x)[0] for x in mytexts]
def train(data_clas):
	print("Building the text classifier")
	learn = text_classifier_learner(data_clas, drop_mult=my_drop_mult)
	learn.load_encoder("../../languageModel/models/"+ 'LM_selfData')
	print("Language model loaded!")
	#learn.freeze()
	learn.freeze_to(-1)
	lr = my_lr/2.0
	learn.fit_one_cycle(1, slice(lr/100, lr))
	#learn.freeze_to(-2)
	#learn.fit(1, slice(lr/100, lr))
	learn.unfreeze()
	lr = my_lr
	learn.fit(my_epochs, slice(lr/100, lr))
	#print("saving the classifier...")
	#learn.save('TC_LM_selfData_fold'+ str(fold))
	return learn

print("Preparing the classifier and experiments using 3-fold CV")
test_df = pd.DataFrame( {'label':  labels_test.astype(str), 'text':  texts_test})
skf = StratifiedKFold(n_splits=3)
for train_index, val_index in skf.split(texts_train_valid, labels_train_valid):
	print("\n\n\nFOLD")
	print(train_index[0:10])
	print(val_index[0:10])
	print("train:", len(train_index), "validation:", len(val_index))
	texts_train, texts_valid= np.asarray(texts_train_valid)[train_index], np.asarray(texts_train_valid)[val_index]
	labels_train, labels_valid = np.asarray(labels_train_valid)[train_index], np.asarray(labels_train_valid)[val_index]
	#print("\n\n A sample of classifier training data:")
	#print(texts_train[0:5])
	#print("\n\n A sample of classifier validation data:")
	#print(texts_valid[0:5])
	train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
	valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
	print("\n\n A sample of classifier training data:")
	print(train_df.head(5))
	print("\n\n A sample of classifier validation data:")
	print(valid_df.head(5))
	data_clas = TextClasDataBunch.from_df(path + "/classification", train_df = train_df, valid_df = valid_df, vocab=data_lm.train_ds.vocab, bs=32)

	learn = train(data_clas)

	#Predictions
	sentence = "Hilary Clinton won the 2016 US election"
	p = learn.predict(sentence)
	print("Prediction for sentence \" " + sentence + "\" is " + str(p))

	print("Results on training data:")
	y = predict(learn, texts_train)
	print(classification_report(labels_train,  list(map(int, y))))

	print("Results on validation data:")
	y = predict(learn, texts_valid)
	print(classification_report( labels_valid, list(map(int, y))))


print("REFIT on train+valid and then testing...")
train_df = pd.DataFrame( {'label':  labels_train_valid.astype(str), 'text':  texts_train_valid})
test_df = pd.DataFrame( { 'label':  labels_test.astype(str), 'text':  texts_test})
data_clas = TextClasDataBunch.from_df(path + "/classification", train_df = train_df, valid_df = test_df, vocab=data_lm.train_ds.vocab, bs=32)

learn = train(data_clas)

#Predictions
print("Results on test data:")
y = predict(learn, texts_test)
print(classification_report( labels_test, list(map(int, y))))

# Data sources used for testing:
texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_d$
print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked , 40, [2,5])
y = predict(learn, texts_test)
print(classification_report( labels_test, list(map(int, y))))


