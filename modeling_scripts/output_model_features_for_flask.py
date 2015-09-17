__author__ = 'erindooley'
import pandas as pd
import pickle

df2 = pd.DataFrame(pickle.load(open('data_files/clean_recent_data.pkl', 'rb')))


df2['primary_focus_area'] = pd.get_dummies(df2.primary_focus_area)
df2['total_price_excluding_optional_support'] = df2['total_price_excluding_optional_support'].map(lambda x: x+1)
df2['total_price_including_optional_support'] = df2['total_price_including_optional_support'].map(lambda x: x+1)
df2['students_reached'] = df2['students_reached'].map(lambda x: x+1)
df2['students_reached'] = df2['students_reached'].replace("nan", 1)

df2 = df2[['primary_focus_area', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'RESP']]

features = df2.columns.values.tolist()
for feature in features:
    per = (len(df2[feature]) - df2[feature].count())/len(df2[feature])
    if per > 0.0:
        print feature, "has %0.2f percent missing values" % (per*100)

print "Pickling"
with open('prediction_app/dummied_data.pkl', 'w') as picklefile:
    pickle.dump(df2, picklefile)
print "DONE"