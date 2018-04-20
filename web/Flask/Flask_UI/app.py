#import svm_prediction
import os
import csv
import json
import sys
sys.path.append('../../../source/')

ROOT = '/home/ling-discourse-lab/Fatemeh/fake_news/'
fact_model_path = ROOT + 'output/intermediate_output/models/tf-idf_SGD_misinformation_binary.pkl'
genre_model_path = ROOT + 'output/intermediate_output/models/tf-idf_SGD_rashkin.pkl'

# Load models
fact_model = TfIDF_model(fact_model_path)
genre_model = TfIDF_model(genre_model)

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
        print("You selected the fact-checking function")
        prediction = fact_model.predict(text)
        label = 'your text is mostly based on facts :)' if prediction == 1 else 'your text is mostly based on imagination :('
    elif selected_model == "genre":
        print("You selected the genre inspection function")
        prediction = genre_model.predict(text)
        label = 'your text looks like hoax' if prediction == 1 else "your text looks like propaganda"
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

