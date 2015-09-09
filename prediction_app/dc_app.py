import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
from patsy import dmatrices
from sklearn import preprocessing

#---------- MODEL IN MEMORY ----------------#

patients = pd.DataFrame(pickle.load(open('dummied_data.pkl', 'rb')))
patients.columns=['price_ex',
 'price_in',
 'students',
 'RESP',
 'Applied Learning',
 'Health & Sports',
 'History & Civics',
 'Literacy & Language',
 'Math & Science',
 'Music & The Arts',
 'Special Needs']


X = patients[['price_ex',
 'price_in',
 'students',
 'Applied Learning',
 'Health & Sports',
 'History & Civics',
 'Literacy & Language',
 'Math & Science',
 'Music & The Arts',
 'Special Needs']]
Y = patients['RESP']
PREDICTOR = LogisticRegression().fit(X,Y)



#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page
    """
    with open("dc_prediction.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 5000 (change to 80 when loaded to AWS?)
# (The default website port)
app.run(host='0.0.0.0', port=5000)
