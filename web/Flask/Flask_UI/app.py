#import svm_prediction
import os
import csv
import json
import sys
import csv

from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime

from flask import Flask
from flask import render_template, request, jsonify, send_from_directory


########### FOR RUNNING ON MY COMPUTER #############

'''

sys.path.append('../../../web/')
from src.textutils import DataLoading
from src.parsingdebunking.DebunkingParser import DebunkingParser
from src.parsingdebunking.data_cleaner import data_clean

#ROOT = '/Users/fa/workspace/temp/web_interface/web/'
ROOT = '~/workspace/shared/sfu/fake_news/web/'

'''

########## REPLACE THE ABOVE SNIPPET WITH THE FOLLOWING FOR RUNNING ON THE SERVER:


sys.path.append('../../../web/')
from src.textutils import DataLoading
from src.parsingdebunking.DebunkingParser import DebunkingParser
from src.parsingdebunking.data_cleaner import data_clean

ROOT = '/home/ftorabia/workspace/shared/sfu/fake_news/web/'



# Load models
print("Loading models...")
genera_model = pickle.load(open(ROOT + "models/tf-idf_SGD_genera_4-way.pkl", 'rb'), encoding='latin1')  #Py27 version: pickle.load(open(ROOT + "models/tf-idf_SGD_genera_4-way.pkl", 'rb'))
genera_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_genera_4-way.pkl", 'rb'),  encoding='latin1')


fact_model = pickle.load(open(ROOT + "models/tf-idf_SGD_misinformation_binary.pkl", 'rb'),  encoding='latin1')
fact_vectorizer = pickle.load(open(ROOT + "models/tf-idf_vectorizer_misinformation_binary.pkl", 'rb'), encoding='latin1')




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


@app.route("/download_data", methods=["GET", "POST"])
def download_data():
    directory ='temporary_files/'
    now = datetime.datetime.now()
    time_label = "_".join([str(x) for x in \
        [now.month, now.day, now.hour, now.minute, now.second]])
    
    #create log file
    log_dir = "logs/"
    log_name = "parsing_request_logs.csv"
    time_stamp = "-".join([str(x) for x in [now.year, now.month, now.day]])
    if not os.path.isfile(log_dir + log_name):
        header = "webname,to_parse_whole_website,to_parse_orginsite,to_clean_data" \
         + ",timestamp, debunking_address\n"
        with open(log_dir + log_name, 'w') as f:
            f.write(header)
    if request.method == "POST":
        # get the information from the front end
        webname = request.values.get("webname")
        is_whole_web = request.values.get("is_whole_web")
        otherReqs = request.values.getlist("otherReqs")
        to_parse_orginsite = ("originSites" in otherReqs)
        to_clean_data = ("clean" in otherReqs)
        Addr = ""
        log_info = [webname, is_whole_web, str(to_parse_orginsite), \
            str(to_clean_data), time_stamp, Addr]
        
        if is_whole_web == "False":
            Addr = request.values.get(webname + "Addr")
            log_info[-1] = Addr
            if not Addr:
                return "No debunking website address provided"
            parser = DebunkingParser(webname)
            parsed_file_name = parser.parsing_web(time_label, Addr, to_parse_orginsite, directory)
            if parsed_file_name == "":
                return send_from_directory(directory=directory, \
                filename="empty.csv", \
                as_attachment=True)
            if to_clean_data:
                parsed_file_name = data_clean(parsed_file_name, directory,\
                 webname, to_parse_orginsite)
            
            with open(log_dir + log_name, 'a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                try:
                    csv_writer.writerow(log_info)
                except Exception as e:
                    print(e)
            return send_from_directory(directory=directory, \
                filename=parsed_file_name, \
                as_attachment=True)
        else:
            #write the log information
            with open(log_dir + log_name, 'a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                try:
                    csv_writer.writerow(log_info)
                except Exception as e:
                    print(e)

            directory = "large_files"
            phase = "phase1"
            if to_parse_orginsite:
                phase = "phase2"
            clean = "raw"
            if to_clean_data:
                clean = "clean"
            
            import glob
            partial_file_name = "_".join([webname, phase, clean]) + "*.zip"
            filename = glob.glob("large_files/" + partial_file_name)[0].split("/")[-1]
            #print(filename)
            return send_from_directory(directory=directory, \
                filename=filename,as_attachment=True)

if __name__ == '__main__':
    app.run(host='localhost', port=7075, debug=True)

