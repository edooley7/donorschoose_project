from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

df = pd.DataFrame(pickle.load(open('../data_files/cleaned_data_with_features.pkl', 'rb')))
print "Done reading dataframe"

df2 = df[df['year'] >= 2014]
print "Done pulling just projects since 2014"

print "Pickling"
with open('../data_files/clean_recent_data.pkl', 'w') as picklefile:
    pickle.dump(df2, picklefile)
print "DONE"
