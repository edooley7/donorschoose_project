import flask
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
from readcalc import readcalc
from textblob import TextBlob
import os
pd.options.mode.chained_assignment = None


#---------- MODEL IN MEMORY ----------------#

df = pd.read_csv("static/merged_data.csv")
completed_df = df[df.RESP == 1]
essay_df = df[['_projectid', ' essay']]

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
PREDICTOR = DecisionTreeClassifier(max_depth = 8, class_weight = "auto" ).fit(X, Y)

completed_df['features_together'] = zip(completed_df['primary_focus_subject_Applied Sciences'],
completed_df['primary_focus_subject_Character Education'],
completed_df['primary_focus_subject_Civics & Government'],
completed_df['primary_focus_subject_College & Career Prep'],
completed_df['primary_focus_subject_Community Service'],
completed_df['primary_focus_subject_ESL'],
completed_df['primary_focus_subject_Early Development'],
completed_df['primary_focus_subject_Economics'],
completed_df['primary_focus_subject_Environmental Science'],
completed_df['primary_focus_subject_Extracurricular'],
completed_df['primary_focus_subject_Financial Literacy'],
completed_df['primary_focus_subject_Foreign Languages'],
completed_df['primary_focus_subject_Gym & Fitness'],
completed_df['primary_focus_subject_Health & Life Science'],
completed_df['primary_focus_subject_Health & Wellness'],
completed_df['primary_focus_subject_History & Geography'],
completed_df['primary_focus_subject_Literacy'],
completed_df['primary_focus_subject_Literature & Writing'],
completed_df['primary_focus_subject_Mathematics'],
completed_df['primary_focus_subject_Music'],
completed_df['primary_focus_subject_Nutrition'],
completed_df['primary_focus_subject_Other'],
completed_df['primary_focus_subject_Parent Involvement'],
completed_df['primary_focus_subject_Performing Arts'],
completed_df['primary_focus_subject_Social Sciences'],
completed_df['primary_focus_subject_Special Needs'],
completed_df['primary_focus_subject_Team Sports'],
completed_df['primary_focus_subject_Visual Arts'],
completed_df['grade_level_Grades 3-5'],
completed_df['grade_level_Grades 6-8'],
completed_df['grade_level_Grades 9-12'],
completed_df['grade_level_Grades PreK-2'],
completed_df['resource_type_Books'],
completed_df['resource_type_Other'],
completed_df['resource_type_Supplies'],
completed_df['resource_type_Technology'],
completed_df['resource_type_Trips'],
completed_df['resource_type_Visitors'])


#---------- URLS AND WEB PAGES -------------#
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open("dc_prediction.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/score", methods=["POST"])
def score():
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    a = data["example"][9:37]
    b = data["example"][41:45]
    c = data["example"][48:54]
    short_x = np.concatenate((a,b,c))
    short_x = tuple(short_x)
    ind = pd.DataFrame(completed_df[' essay'].loc[completed_df['features_together'] == short_x])
    text = ind[' essay'].values[-1]
    calc = readcalc.ReadCalc(text)
    polarity = TextBlob(text).polarity
    polarity = round(polarity*10,1)
    user_readability = round(calc.get_ari_index(),1)
    results = {"score": score[0][1], "project_text": text, "user_readability": user_readability, "neighbor_polarity": polarity }
    return flask.jsonify(results)

@app.route("/grade_essay", methods=["POST"])
def grade_essay():
    data = flask.request.json
    calc = readcalc.ReadCalc(data["essay"])
    ari_score = round(calc.get_ari_index(),2)
    polarity = TextBlob(data["essay"]).polarity
    polarity = round(polarity*10,1)
    results = {"readability": ari_score, "polarity": polarity }
    return flask.jsonify(results)
    
#--------- RUN WEB APP SERVER ------------#
#app.run(host='0.0.0.0', port=80)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug = True)
  