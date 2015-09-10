from __future__ import division
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.cross_validation import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

df = pd.DataFrame(pickle.load(open('data_files/dummied_recent_data.pkl', 'rb')))
print "Done reading dataframe"

X = df[[
    'school_charter',
    'school_magnet',
    'school_year_round',
    'school_nlns',
    'school_kipp',
    'school_charter_ready_promise',
    'teacher_teach_for_america',
    'teacher_ny_teaching_fellow',
    'vendor_shipping_charges',
    'total_price_including_optional_support',
    'students_reached',
    'month',
    'quarter',
    'year',
    'optional_support',
    'school_previous_projects',
    'teacher_previous_projects',
    'log_price_including',
    'sqrt_students_reached',
    'price_per_student',
    'total_state_donors',
    'total_state_projects',
    'state_avg_donors',
    'primary_focus_area_Applied Learning',
    'primary_focus_area_Health & Sports',
    'primary_focus_area_History & Civics',
    'primary_focus_area_Literacy & Language',
    'primary_focus_area_Math & Science',
    'primary_focus_area_Music & The Arts',
    'primary_focus_area_Special Needs',
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

std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
print "Done scaling"

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)
print "Done splitting"


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Soft Voting/Majority Rule classifier for unfitted clfs.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of classifiers.
      Invoking the `fit` method on the `VotingClassifier` will fit clones
      of those original classifiers that will be stored in the class attribute
      `self.clfs_`.

    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of
      the sums of the predicted probalities, which is recommended for
      an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) to weight the occurances of
      predicted class labels (`hard` voting) or class probabilities
      before averaging (`soft` voting). Uses uniform weights if `None`.

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """

    def __init__(self, clfs, voting='hard', weights=None):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        """ Fit the clfs.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output' \
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = []
        for clf in self.clfs:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.clfs_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                lambda x:
                np.argmax(np.bincount(x,
                                      weights=self.weights)),
                axis=1,
                arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """ Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])


clf1 = BernoulliNB()
clf2 = GaussianNB()
clf3 = RandomForestClassifier()
#clf4 = GradientBoostingClassifier(n_estimators=500)
clf5 = AdaBoostClassifier()
clf6 = LogisticRegression()
clf7 = DecisionTreeClassifier()

np.random.seed(123)
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3, clf5, clf6, clf7], voting='soft')

for clf, label in zip([clf1, clf2, clf3, clf5, clf6, clf7, eclf], ['Bernoulli', 'Gaussian', 'Random Forest', 'ADA', 'Log', 'DTC', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='precision')
    print("Precision: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
