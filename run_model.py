from __future__ import division
import pandas as pd
import numpy as np
from patsy import dmatrices
pd.options.mode.chained_assignment = None
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.cross_validation import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.DataFrame(pickle.load(open('clean_recent_data.pkl', 'rb')))
print "Done reading dataframe"


y, X = dmatrices(
    'RESP ~ teacher_previous_projects + school_previous_projects + primary_focus_area + primary_focus_subject + np.log(total_price_including_optional_support + np.sqrt(students_reached))',
    data=df, return_type='dataframe')

y = np.ravel(y)

std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
print "Done scaling"

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)
print "Done splitting"

def always_complete(x):
    return [1] * len(x)
y_pred = always_complete(X)

print "Baseline = %0.2f" % accuracy_score(y, y_pred)

def test_model(name, main_model):
    model = main_model
    model.fit(x_train, y_train)
    precision = np.mean(cross_val_score(main_model, X, y.ravel(), scoring = 'precision'))
    print name, "precision", precision
    print name, "confusion matrix", confusion_matrix(y_test, model.predict(x_test))

test_model("Logistic Regression", LogisticRegression())
test_model("GaussianNB", GaussianNB())
#test_model("MultinomialNB", MultinomialNB())
test_model("DecisionTreeClassifier", DecisionTreeClassifier())
test_model("RandomForestClassifier", RandomForestClassifier())
test_model("GradientBoostingClassifier", GradientBoostingClassifier())
test_model("KNeighborsClassifier", KNeighborsClassifier())
test_model("SVC", SVC())
print "DONE"
