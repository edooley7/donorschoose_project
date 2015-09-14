import flask
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree

#---------- MODEL IN MEMORY ----------------#

df = pd.DataFrame(pickle.load(open('static/dummied_recent_data.pkl', 'rb')))
essay_df = pd.read_csv("static/cleaned_essays.csv")

X = df[[
 'school_previous_projects',
 'teacher_previous_projects',
 'month',
 'log_price_including',
 'sqrt_students_reached',
 'price_per_student',
 'total_state_donors',
 'total_state_projects',
 'state_avg_donors',
 'primary_focus_subject_Applied Sciences',
 'primary_focus_subject_Character Education',
 'primary_focus_subject_Civics & Government',
 'primary_focus_subject_College & Career Prep',
 'primary_focus_subject_Community Service',
 'primary_focus_subject_ESL',
 'primary_focus_subject_Early Development',
 'primary_focus_subject_Economics',
 'primary_focus_subject_Environmental Science',
 'primary_focus_subject_Extracurricular',
 'primary_focus_subject_Financial Literacy',
 'primary_focus_subject_Foreign Languages',
 'primary_focus_subject_Gym & Fitness',
 'primary_focus_subject_Health & Life Science',
 'primary_focus_subject_Health & Wellness',
 'primary_focus_subject_History & Geography',
 'primary_focus_subject_Literacy',
 'primary_focus_subject_Literature & Writing',
 'primary_focus_subject_Mathematics',
 'primary_focus_subject_Music',
 'primary_focus_subject_Nutrition',
 'primary_focus_subject_Other',
 'primary_focus_subject_Parent Involvement',
 'primary_focus_subject_Performing Arts',
 'primary_focus_subject_Social Sciences',
 'primary_focus_subject_Special Needs',
 'primary_focus_subject_Team Sports',
 'primary_focus_subject_Visual Arts',
 'poverty_level_high poverty',
 'poverty_level_highest poverty',
 'poverty_level_low poverty',
 'poverty_level_moderate poverty',
 'grade_level_Grades 3-5',
 'grade_level_Grades 6-8',
 'grade_level_Grades 9-12',
 'grade_level_Grades PreK-2',
 'school_metro_rural',
 'school_metro_suburban',
 'school_metro_urban',
 'resource_type_Books',
 'resource_type_Other',
 'resource_type_Supplies',
 'resource_type_Technology',
 'resource_type_Trips',
 'resource_type_Visitors']]
Y = df[['RESP']]
Y = np.ravel(Y)
PREDICTOR = LogisticRegression().fit(X, Y)
#PREDICTOR = DecisionTreeClassifier(max_depth = 4, class_weight = "auto" ).fit(X, Y)


lookup = df[[
 '_projectid',
 'school_previous_projects',
 'teacher_previous_projects',
 'month',
 'log_price_including',
 'sqrt_students_reached',
 'price_per_student',
 'total_state_donors',
 'total_state_projects',
 'state_avg_donors',
 'primary_focus_subject_Applied Sciences',
 'primary_focus_subject_Character Education',
 'primary_focus_subject_Civics & Government',
 'primary_focus_subject_College & Career Prep',
 'primary_focus_subject_Community Service',
 'primary_focus_subject_ESL',
 'primary_focus_subject_Early Development',
 'primary_focus_subject_Economics',
 'primary_focus_subject_Environmental Science',
 'primary_focus_subject_Extracurricular',
 'primary_focus_subject_Financial Literacy',
 'primary_focus_subject_Foreign Languages',
 'primary_focus_subject_Gym & Fitness',
 'primary_focus_subject_Health & Life Science',
 'primary_focus_subject_Health & Wellness',
 'primary_focus_subject_History & Geography',
 'primary_focus_subject_Literacy',
 'primary_focus_subject_Literature & Writing',
 'primary_focus_subject_Mathematics',
 'primary_focus_subject_Music',
 'primary_focus_subject_Nutrition',
 'primary_focus_subject_Other',
 'primary_focus_subject_Parent Involvement',
 'primary_focus_subject_Performing Arts',
 'primary_focus_subject_Social Sciences',
 'primary_focus_subject_Special Needs',
 'primary_focus_subject_Team Sports',
 'primary_focus_subject_Visual Arts',
 'poverty_level_high poverty',
 'poverty_level_highest poverty',
 'poverty_level_low poverty',
 'poverty_level_moderate poverty',
 'grade_level_Grades 3-5',
 'grade_level_Grades 6-8',
 'grade_level_Grades 9-12',
 'grade_level_Grades PreK-2',
 'school_metro_rural',
 'school_metro_suburban',
 'school_metro_urban',
 'resource_type_Books',
 'resource_type_Other',
 'resource_type_Supplies',
 'resource_type_Technology',
 'resource_type_Trips',
 'resource_type_Visitors']]
 
collist = ['school_previous_projects',
 'teacher_previous_projects',
 'month',
 'log_price_including',
 'sqrt_students_reached',
 'price_per_student',
 'total_state_donors',
 'total_state_projects',
 'state_avg_donors',
 'primary_focus_subject_Applied Sciences',
 'primary_focus_subject_Character Education',
 'primary_focus_subject_Civics & Government',
 'primary_focus_subject_College & Career Prep',
 'primary_focus_subject_Community Service',
 'primary_focus_subject_ESL',
 'primary_focus_subject_Early Development',
 'primary_focus_subject_Economics',
 'primary_focus_subject_Environmental Science',
 'primary_focus_subject_Extracurricular',
 'primary_focus_subject_Financial Literacy',
 'primary_focus_subject_Foreign Languages',
 'primary_focus_subject_Gym & Fitness',
 'primary_focus_subject_Health & Life Science',
 'primary_focus_subject_Health & Wellness',
 'primary_focus_subject_History & Geography',
 'primary_focus_subject_Literacy',
 'primary_focus_subject_Literature & Writing',
 'primary_focus_subject_Mathematics',
 'primary_focus_subject_Music',
 'primary_focus_subject_Nutrition',
 'primary_focus_subject_Other',
 'primary_focus_subject_Parent Involvement',
 'primary_focus_subject_Performing Arts',
 'primary_focus_subject_Social Sciences',
 'primary_focus_subject_Special Needs',
 'primary_focus_subject_Team Sports',
 'primary_focus_subject_Visual Arts',
 'poverty_level_high poverty',
 'poverty_level_highest poverty',
 'poverty_level_low poverty',
 'poverty_level_moderate poverty',
 'grade_level_Grades 3-5',
 'grade_level_Grades 6-8',
 'grade_level_Grades 9-12',
 'grade_level_Grades PreK-2',
 'school_metro_rural',
 'school_metro_suburban',
 'school_metro_urban',
 'resource_type_Books',
 'resource_type_Other',
 'resource_type_Supplies',
 'resource_type_Technology',
 'resource_type_Trips',
 'resource_type_Visitors']

tree = KDTree(lookup[collist], metric = "chebyshev")



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
    # data["example"] needs to be length 54
    #x = np.matrix("5.270380747855014,2.23606797749979,38.898,334846.0,85457.0,3.918298091437799,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,22.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0]")
    x = np.matrix(data["example"])
    print data["example"]
    score = PREDICTOR.predict_proba(x)
    dist, ind = tree.query(data["example"], k=1)
    new = lookup.reset_index(drop=True)
    proj_id = new.ix[ind[0][0]]['_projectid']
    print proj_id
    essay_text = pd.DataFrame(essay_df[' essay'].loc[essay_df['_projectid'] == proj_id])
    print essay_text
    text = essay_text[' essay'].values[0]
    user_readability = len(text)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0][1], "project_text": text, "user_readability": user_readability }
    return flask.jsonify(results)

@app.route("/grade_essay", methods=["POST"])
def grade_essay():
    # Get decision score for our example that came with the request
    data = flask.request.json
    #print data["example"]
    print data["essay"]
    essay_len = len(data["essay"])
    print essay_len
    results = {"readability": essay_len}
    return flask.jsonify(results)
    
#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 5000 (change to 80 when loaded to AWS?)
# (The default website port)
app.run(host='0.0.0.0', port=5000)
