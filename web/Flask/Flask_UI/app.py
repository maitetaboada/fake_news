#import svm_prediction
import os
import csv
import json
import sys
import csv

from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer





sys.path.append('../../../web/')
from src.textutils import DataLoading

#ROOT = '/Users/fa/workspace/temp/web_interface/web/'
ROOT = '/Users/ftorabia/workspace/shared/sfu/fake_news/web/'



########## REPLACE THE ABOVE SNIPPET WITH THE FOLLOWING FOR RUNNING ON THE SERVER:
'''

sys.path.append('../../src/')
from textutils import DataLoading

ROOT = '/home/ling-discourse-lab/Fatemeh/web_interface/web/'

'''








from flask import Flask
from flask import render_template, request, jsonify

# Load models
print("Loading models...")
genera_model = pickle.load(open(ROOT + "models/tf-idf_SGD_genera_4-way.pkl", 'rb'))
genera_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_genera_4-way.pkl", 'rb'))


fact_model = pickle.load(open(ROOT + "models/tf-idf_SGD_misinformation_binary.pkl", 'rb'))
fact_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_misinformation_binary.pkl", 'rb'))




#### LOAD and REUSE ###
print("Extracting features from the test data using the same vectorizer")
#text = "Clinton election day president 2016 2015    "
text = "GREEN BAY, WIDavid Horsted, 45, announced Monday that he's seen a whole heck of a lot during his 20 years driving a taxi. 'Aw, geez, the people I've met and the places I've seenthe stories would make your head spin,' Horsted said. 'I've been from Lambeau Field to the Barhausen Waterfowl Preserve and every place in between. One time, one of the Packers even threw up in my cab, but I don't think I should say who.' With a little prodding, Horsted said the person's first name rhymes with 'baloney' and last name with 'sandwich.'"
text = BeautifulSoup(text)
text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))


X_test = fact_vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
pred = fact_model.predict(X_test)
conf = fact_model.decision_function(X_test)
#prob = fact_model.predict_proba(X_test)
print("Prediction for this input is:")
print(pred)
print("Confidence for this input is:")
print(conf)
#print("Prediction probability:")
#print(prob)



X_test = genera_vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
pred = genera_model.predict(X_test)
conf = genera_model.decision_function(X_test)
#prob = genera_model.predict_proba(X_test)
print("Prediction for this input is:")
print(pred)
print("Confidence for this input is:")
print(conf)
#print("Prediction probability:")


#################################### ACTUAL FLASK CODE ##############################################

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_prediction", methods=["GET", "POST"])
def get_prediction():
    text = request.args.get("text")
    selected_model = request.args.get("model")
    label = 'Default'
    conf = [0, 0]
    axis = ["",""]
    print("SELECTED MODEL: ", selected_model)
    if selected_model == "fact":

        text = BeautifulSoup(text)
        text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))
        print("You selected the fact-checking function")
        X_test = fact_vectorizer.transform([text])
        pred = fact_model.predict(X_test)
        print("Prediction for this input is:")
        print(pred)
        label = 'text is mostly based on FACTS' if pred == 0 else 'text is mostly based on FALSE INFORMATION'
        conf = [(fact_model.decision_function(X_test)[0])]
        print(label)
        print(conf)
        axis = ["Confidence"]
        #label = label + " with confidence score of " + str(fact_model.predict_proba(X_test))

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
        conf = (genera_model.decision_function(X_test)[0]).tolist()
        label = 'text looks like ' + label.upper()
        axis = ["Trusted", "Propaganda", "Satire", "Hoax"]
        #label = 'Text looks like ' + label + " with confidence score of " + str(
        #    genera_model.predict_proba(X_test))

    else:
        print("Please select a function first.")
        return jsonify(predicted_label="Please select a function first!", chart_values=conf, chart_labels=axis)
        #return(render_template("index.html", predicted_label="Please select a function first!", chart_values=conf, chart_labels=axis))
    print(text)
    return jsonify(predicted_label="According to our model, " + label + ".", chart_values=conf, chart_labels=axis)
    #return (render_template("index.html", predicted_label="According to our model: " + label.upper() + ".", chart_values=conf, chart_labels=axis))

@app.route("/select_model", methods=["GET", "POST"])
def select_model():
    model = request.args.get('result')

    print("THIS IS THE MODEL: ", model)
    return jsonify(resp='You selected: ' + model)


@app.route("/get_feedback", methods=["GET", "POST"])
def get_feedback():
    text = request.args.get('text')
    label = request.args.get('answer')
    comments = request.args.get('comments')
    print([text, label, comments])
    file_exists = os.path.isfile('feedback.csv')
    print("CSV file existance" + str(file_exists))
    csvfile = open('feedback.csv', 'a+')
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(['Text', 'Label', 'Comments'])
    writer.writerow([text, label, comments])
    print("Added to feedbacks!")
    return jsonify(feedback='Thank you for your feedback!')


if __name__ == '__main__':
    app.run(host='localhost', port=7070)

