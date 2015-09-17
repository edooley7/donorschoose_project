from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

recent_df = pd.DataFrame(pickle.load(open('../data_files/clean_recent_data.pkl', 'rb')))
print "Done reading recent dataframe"
# full_df = pd.DataFrame(pickle.load(open('../data_files/cleaned_data_with_features.pkl', 'rb')))
# print "Done reading full dataframe"

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

# dummied_full_df = dummy_variables(full_df, df_type_dict)[0]
# print "Done dummy-ing full data"
dummied_recent_df = dummy_variables(recent_df, df_type_dict)[0]
dummied_recent_df = dummied_recent_df[[
    '_projectid',
    'school_previous_projects',
    'teacher_previous_projects',
    'month',
    'log_price_including',
    'sqrt_students_reached',
    'price_per_student',
    'total_state_donors',
    'total_state_projects',
    'state_avg_donors',
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
    'resource_type_Books',
    'resource_type_Other',
    'resource_type_Supplies',
    'resource_type_Technology',
    'resource_type_Trips',
    'resource_type_Visitors',
    'RESP']]
print "Done dummy-ing recent data"

print "Pickling recent data"
with open('../data_files/dummied_recent_data.pkl', 'w') as picklefile:
    pickle.dump(dummied_recent_df, picklefile)
# print "Done pickling recent data"
# print "Pickling full data"
# with open('../data_files/dummied_full_data.pkl', 'w') as picklefile:
#    pickle.dump(dummied_full_df, picklefile)
print "DONE"
