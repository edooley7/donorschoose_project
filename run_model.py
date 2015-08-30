from __future__ import division
import pandas as pd
import numpy as np
from patsy import dmatrices

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("~/donorschoose/cleaned_dataframe.csv")
print "Done reading dataframe"

y, X = dmatrices(
    'RESP ~ school_state+school_charter+school_magnet+ school_year_round+ school_nlns+ school_kipp+ school_charter_ready_promise+ teacher_prefix+teacher_teach_for_america+teacher_ny_teaching_fellow+primary_focus_subject+primary_focus_area+resource_type+poverty_level+grade_level+total_price_including_optional_support+students_reached+optional_support+quarter+teacher_per_success+school_per_success',
    data=df, return_type='dataframe')

y = np.ravel(y)

std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
print "Done scaling"


def always_complete(x):
    return [1] * len(x)
y_pred = always_complete(X)

print "Baseline = %0.2f" % accuracy_score(y, y_pred)

names, accs = [], []
for algorithm in (LogisticRegression,
                  KNeighborsClassifier,
                  GaussianNB,
                  SVC,
                  DecisionTreeClassifier,
                  RandomForestClassifier):
    accuracy = np.mean(cross_val_score(algorithm(), X, y, cv=10))
    print '%-30s %0.4f' % (algorithm.__name__, accuracy)
    names.append(algorithm.__name__)
    accs.append(accuracy)
print "DONE"