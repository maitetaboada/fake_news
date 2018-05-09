#import svm_prediction
import os
import csv
import json
import sys

#sys.path.append('../../src/')
sys.path.append('../../../')

from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from web.src.textutils import DataLoading




ROOT = '/home/ling-discourse-lab/Fatemeh/web_interface/web/'
#ROOT = '/Users/fa/workspace/shared/sfu/fake_news/web/'



# Load models
fact_model = pickle.load(open(ROOT + "models/tf-idf_SGD_misinformation_binary.pkl", 'rb'))
fact_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_misinformation_binary.pkl", 'rb'))

genera_model = pickle.load(open(ROOT + "models/tf-idf_SGD_genera_4-way.pkl", 'rb'))
genera_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_genera_4-way.pkl", 'rb'))


#### LOAD and REUSE ###
print("\n\nLOAD and REUSE\n\n")
print("Extracting features from the test data using the same vectorizer")
#text = "Clinton election day president 2016 2015    "
text = "GREEN BAY, WIDavid Horsted, 45, announced Monday that he's seen a whole heck of a lot during his 20 years driving a taxi. 'Aw, geez, the people I've met and the places I've seenthe stories would make your head spin,' Horsted said. 'I've been from Lambeau Field to the Barhausen Waterfowl Preserve and every place in between. One time, one of the Packers even threw up in my cab, but I don't think I should say who.' With a little prodding, Horsted said the person's first name rhymes with 'baloney' and last name with 'sandwich.'"
text = BeautifulSoup(text)
text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))


X_test = fact_vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
pred = fact_model.predict(X_test)
conf = fact_model.decision_function(X_test)
print("Prediction for this input is:")
print(pred)
print("Confidence for this input is:")
print(conf)


X_test = genera_vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
pred = genera_model.predict(X_test)
conf = genera_model.decision_function(X_test)
print("Prediction for this input is:")
print(pred)
print("Confidence for this input is:")
print(conf)



from flask import Flask
from flask import render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_prediction", methods=["GET", "POST"])
def get_prediction():
    text = request.args.get("result")
    selected_model = request.args.get("model")
    label = 'Default'
    print("SELECTED MODEL: ", selected_model)
    if selected_model == "fact":
        text = BeautifulSoup(text)
        text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))
        print("You selected the fact-checking function")
        X_test = fact_vectorizer.transform([text])
        pred = fact_model.predict(X_test)
        print("Prediction for this input is:")
        print(pred)
        label = 'Text is mostly based on facts!' if pred == 0 else 'Text is mostly based on false information!'
        label = label + " with confidence score of " + str(fact_model.decision_function(X_test)[0])
    elif selected_model == "genera":
        print("You selected the genre inspection function")
        transdict = {
            2: "Satire",
            3: "Hoax",
            1: "Propaganda",
            0: "Truested"
        }
        X_test = genera_vectorizer.transform([text])
        pred = genera_model.predict(X_test)
        print("Prediction for this input is:")
        print(pred)
        pred = genera_model.predict(X_test)[0]
        label = transdict[pred]
        label = 'Text looks like ' + label + " with confidence score of " + str(genera_model.decision_function(X_test)[0])
    else:
        print("Please select a function first.")
        return jsonify(predicted_label="Please select a function first!")
    print(text)
    return jsonify(predicted_label="According to our model: " + label.upper() + ".")

@app.route("/select_model", methods=["GET", "POST"])
def select_model():
    model = request.args.get('result')

    print("THIS IS THE MODEL: ", model)
    return jsonify(resp='You selected: ' + model)


@app.route("/get_feedback", methods=["GET", "POST"])
def feedback():
    text = request.args.get('result')
    label = request.args.get('label')
    comments = request.args.get('comments')

    file_exists = os.path.isfile('feedback.csv')
    with open('feedback.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Text', 'Label', 'Comments'])
        writer.writerow([text, label, comments])
        return jsonify(feedback='Thank you for your feedback!')


if __name__ == '__main__':
    app.run(host='localhost', port=8484)