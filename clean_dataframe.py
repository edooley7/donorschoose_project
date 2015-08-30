from __future__ import division
import pandas as pd


df = pd.read_csv("~/donorschoose/opendata_projects.csv")
print "Done reading csv"

#Throw out reallocated and live projects
df2 = df[df['funding_status'] != "reallocated"]
df2 = df2[df2['funding_status'] != "live"]
print "Done step 1/4"

#Parse dates
df2['date_posted'] = df2['date_posted'].map(lambda x: pd.to_datetime(x))
df2['date_expiration'] = df2['date_expiration'].map(lambda x: pd.to_datetime(x))
df2['date_completed'] = df2['date_completed'].map(lambda x: pd.to_datetime(x))
df2['date_thank_you_packet_mailed'] = df2['date_thank_you_packet_mailed'].map(lambda x: pd.to_datetime(x))
df2['month'] = df2.date_posted.map(lambda x: x.month)
df2['quarter'] = df2.date_posted.map(lambda x: x.quarter)
print "Done step 2/4"

#Feature engineering
df2['time_to_expire'] = df2.date_expiration - df2.date_posted
df2['amount_optional_support'] = df2.total_price_including_optional_support - df2.total_price_excluding_optional_support
df2['optional_support'] = 0
df2['optional_support'][df2['amount_optional_support'] > 0] = 1
df2['per_optional_support'] = df2.amount_optional_support/df2.total_price_excluding_optional_support
print "Done step 3/4"

df2 = df2.sort("date_posted")
df2['teacher_previous_success'] = df2.groupby('_teacher_acctid')['RESP'].cumsum()
df2['school_previous_success'] = df2.groupby('_schoolid')['RESP'].cumsum()
df2['teacher_previous_projects'] = df2.groupby('_teacher_acctid').cumcount()+1
df2['school_previous_projects'] = df2.groupby('_schoolid').cumcount()+1
df2['teacher_per_success'] = df2['teacher_previous_success']/df2['teacher_previous_projects']
df2['school_per_success'] = df2['school_previous_success']/df2['school_previous_projects']
print "Done step 4/4"

print "Writing to csv"
df2.to_csv("~/donorschoose/cleaned_dataframe.csv")
print "DONE"