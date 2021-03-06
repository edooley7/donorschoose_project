from __future__ import division
import numpy as np
from patsy import dmatrices
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydot
from sklearn import tree

import pandas as pd

pd.options.mode.chained_assignment = None
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.cross_validation import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.DataFrame(pickle.load(open('data_files/dummied_recent_data.pkl', 'rb')))
print "Done reading dataframe"

original_X = df[[
 'vendor_shipping_charges',
 'month',
 'quarter',
 'school_previous_projects',
 'teacher_previous_projects',
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
 'teacher_gender_Female',
 'teacher_gender_Male',
 'resource_type_Books',
 'resource_type_Other',
 'resource_type_Supplies',
 'resource_type_Technology',
 'resource_type_Trips',
 'resource_type_Visitors']]
y = df[['RESP']]

y = np.ravel(y)

std_scale = preprocessing.StandardScaler().fit(original_X)
X = std_scale.transform(original_X)
print "Done scaling"

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)
print "Done splitting"

dtc = DecisionTreeClassifier(max_depth = 4, class_weight = "auto" ).fit(x_train, y_train)
dot_data = StringIO()
tree.export_graphviz(dtc, out_file=dot_data, feature_names = original_X.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("trees.pdf")