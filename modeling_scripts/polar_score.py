from __future__ import division
import pandas as pd
import pickle
from textblob import TextBlob

pd.options.mode.chained_assignment = None

df = pd.read_csv("../prediction_app/static/merged_data.csv")
print "done reading csv"

essay_df = df[['_projectid', 'RESP', ' essay' ]]
essay_df['new_essay'] = essay_df[' essay'].map(lambda x: type(x))
essay_df = essay_df[essay_df.new_essay == str]
print "done throwing out floats"
print "percent remaining", len(essay_df)/len(df)
essay_df.new_essay = essay_df[' essay'].map(lambda x: x.decode('utf-8'))
print "done decoding"

essay_df['polar'] = essay_df.new_essay.map(lambda x: TextBlob(x).polarity)
essay_df['subjectivity'] = essay_df.new_essay.map(lambda x: TextBlob(x).subjectivity)


print "Pickling"
with open('../data_files/polar_score.pkl', 'wb') as picklefile:
    pickle.dump(essay_df, picklefile)
print "DONE"
