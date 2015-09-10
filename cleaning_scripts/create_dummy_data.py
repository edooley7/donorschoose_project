from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

recent_df = pd.DataFrame(pickle.load(open('../data_files/clean_recent_data.pkl', 'rb')))
print "Done reading recent dataframe"
full_df = pd.DataFrame(pickle.load(open('../data_files/cleaned_data_with_features.pkl', 'rb')))
print "Done reading full dataframe"

df_type_dict = {'_projectid': 'regular',
                '_teacher_acctid': 'regular',
                '_schoolid': 'regular',
                'school_state': 'regular',
                'school_metro': 'nominal',
                'school_charter': 'regular',
                'school_magnet': 'regular',
                'school_year_round': 'regular',
                'school_nlns': 'regular',
                'school_kipp': 'regular',
                'school_charter_ready_promise': 'regular',
                'teacher_teach_for_america': 'regular',
                'teacher_ny_teaching_fellow': 'regular',
                'primary_focus_subject': 'nominal',
                'primary_focus_area': 'nominal',
                'resource_type': 'nominal',
                'poverty_level': 'nominal',
                'grade_level': 'nominal',
                'vendor_shipping_charges': 'regular',
                'total_price_excluding_optional_support': 'regular',
                'total_price_including_optional_support': 'regular',
                'students_reached': 'regular',
                'date_posted': 'regular',
                'month': 'regular',
                'quarter': 'regular',
                'year': 'regular',
                'time_to_expire': 'regular',
                'optional_support': 'regular',
                'school_previous_projects': 'regular',
                'teacher_previous_projects': 'regular',
                'log_price_including': 'regular',
                'log_price_excluding': 'regular',
                'sqrt_students_reached': 'regular',
                'student_bins': 'nominal',
                'price_in_bins': 'nominal',
                'price_ex_bins': 'nominal',
                'teacher_gender': 'nominal',
                'price_per_student': 'regular',
                'price_per_student_bins': 'nominal',
                'total_state_donors': 'regular',
                'total_state_projects': 'regular',
                'state_avg_donors': 'regular',
                'RESP': 'regular'}


def dummy_variables(data, data_type_dict):
    # Loop over nominal variables.
    for variable in filter(lambda x: data_type_dict[x] == 'nominal',
                           data_type_dict.keys()):

        dummy_df = pd.get_dummies(data[variable], prefix=variable)

        # Add dummy variables to main df.
        data = data.drop(variable, axis=1)
        data = data.join(dummy_df)

    return [data, data_type_dict]


dummied_full_df = dummy_variables(full_df, df_type_dict)[0]
print "Done dummy-ing full data"
dummied_recent_df = dummy_variables(recent_df, df_type_dict)[0]
print "Done dummy-ing recent data"


print "Pickling recent data"
with open('../data_files/dummied_recent_data.pkl', 'w') as picklefile:
    pickle.dump(dummied_recent_df, picklefile)
print "Done pickling recent data"
print "Pickling full data"
with open('../data_files/dummied_full_data.pkl', 'w') as picklefile:
    pickle.dump(dummied_full_df, picklefile)
print "DONE"
