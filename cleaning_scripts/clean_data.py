from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

df = pd.read_csv("~/donorschoose/data_files/opendata_projects.csv", parse_dates = ['date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
print "Done reading csv"

# Pickle dump currently live projects as hold out set for final testing
live_df = df[df['funding_status'] =='live']
with open('holdout_set.pkl', 'w') as live_file:
    pickle.dump(live_df, live_file)
print "Done creating holdout set"

# Throw out reallocated and live projects and label rows
df2 = df[df['funding_status'] != "reallocated"]
df2 = df2[df2['funding_status'] != "live"]
df2['RESP'] = 0
df2['RESP'][df2['funding_status'] == 'completed'] = 1
per_remaining = len(df2) / len(df)
print "Percent of original data remaining: %0.2f" % (per_remaining * 100)
print "Done step 1/5"

# Replace binary features with 0,1
binary_features = ['school_charter',
 'school_magnet',
 'school_year_round',
 'school_nlns',
 'school_kipp',
 'school_charter_ready_promise',
'teacher_teach_for_america',
 'teacher_ny_teaching_fellow']
for feature in binary_features:
    df2[feature] = df2[feature].replace("t", 0)
    df2[feature] = df2[feature].replace("f", 1)
print "Done step 2/5"

# Calculate month, quarter and year for date project was posted on site
df2['month'] = df2.date_posted.map(lambda x: x.month)
df2['quarter'] = df2.date_posted.map(lambda x: x.quarter)
df2['year'] = df2.date_posted.map(lambda x: x.year)
print "Done step 3/5"

# Feature engineering (expiration dates and amount/percentage of optional support)
df2['time_to_expire'] = df2.date_expiration - df2.date_posted
df2['amount_optional_support'] = df2.total_price_including_optional_support - df2.total_price_excluding_optional_support
df2['optional_support'] = 0
df2['optional_support'][df2['amount_optional_support'] > 0] = 1
df2['per_optional_support'] = df2.amount_optional_support / df2.total_price_excluding_optional_support
print "Done step 4/5"

# Feature engineering (previous projects posted by school & teacher)
school_posted = df2.set_index('date_posted').groupby('_schoolid').cumcount()
df2['school_previous_projects'] = school_posted.values

teacher_posted = df2.set_index('date_posted').groupby('_teacher_acctid').cumcount()
df2['teacher_previous_projects'] = teacher_posted.values
print "Done step 5/5"

print "Pickling"
with open('cleaned_data_with_features.pkl', 'wb') as picklefile:
    pickle.dump(df2, picklefile)
print "DONE"