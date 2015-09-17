from __future__ import division
import pandas as pd
import pickle
from readcalc import readcalc

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

essay_df['ari'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_ari_index())
print "done ari"
essay_df['coleman'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_coleman_liau_index())
print "done coleman"
essay_df['flesch_grade'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_flesch_kincaid_grade_level())
print "done flesch grade"
essay_df['flesch_ease'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_flesch_reading_ease())
print "done flesch ease"
essay_df['dale'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_dale_chall_score())
print "done dale"
essay_df['gunning'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_gunning_fog_index())
print "done gunning"
essay_df['lix'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_lix_index())
print "done lix"
essay_df['smog'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_smog_index())
print "done smog"

print "Pickling"
with open('../data_files/read_score.pkl', 'wb') as picklefile:
    pickle.dump(essay_df, picklefile)
print "DONE"
