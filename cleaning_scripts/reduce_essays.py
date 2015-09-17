from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

df = pd.read_csv("~/donorschoose/data_files/cleaned_essays.csv")
print "Done reading essay csv"

recent_df = pd.DataFrame(pickle.load(open('../data_files/dummied_recent_data.pkl', 'rb')))
print "Done reading recent projects file"

merged = recent_df.merge(df, how="left", on="_projectid")
merged = merged[['_projectid',
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
                 'RESP',
                 ' essay']]

print "writing to csv"
merged.to_csv("../prediction_app/static/merged_data.csv", index=False)
print "DONE"
