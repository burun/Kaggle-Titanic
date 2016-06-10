import numpy as np
import pandas as pd
import os

df_train = pd.read_csv(os.path.dirname(__file__) + '/../Data Files/train.csv')
df_test = pd.read_csv(os.path.dirname(__file__) + '/../Data Files/test.csv')

# Get titles from Name column
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

titles_list = df_train.Name.map(lambda x: get_title(x))
# print(titles_list.value_counts())

# Normalize the titles, returning 'Mr', 'Master', 'Miss' or 'Mrs'
def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major', 'Dr']:
        return 'Officer'
    elif title in ['Don', 'Jonkheer', 'Sir']:
        return 'Sir'
    elif title in ['Mme', 'Mrs', 'Ms']:
        return 'Mrs'
    elif title in ['the Countess', 'Lady', 'Dona']:
        return 'Lady'
    elif title in ['Mlle', 'Miss']:
        return 'Miss'
    elif title in ['Mr']:
        return 'Mr'
    else:
        return title

df_train['Title'] = df_train['Name'].map(lambda x: get_title(x))
df_train['Title'] = df_train.apply(replace_titles, axis=1)
print(df_train.Title)
