import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
from patsy import dmatrices
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it
df = pd.DataFrame(pickle.load(open('clean_recent_data.pkl', 'rb')))

y, X = dmatrices(
    'RESP ~ primary_focus_area + primary_focus_subject + np.log(total_price_including_optional_support + np.sqrt(students_reached))',
    data=df, return_type='dataframe')

y = np.ravel(y)

std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
PREDICTOR = LogisticRegression().fit(X,y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, dc_prediction.html
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

# Start the app server on port 5000
# (The default website port)
app.run(host='0.0.0.0', port=5000)
