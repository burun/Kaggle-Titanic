import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Pandas DataFrame
df_train = pd.read_csv('../Data Files/train.csv')
df_test = pd.read_csv('../Data Files/test.csv')

# Data Munging
# Referencing and filtering
# Create Gender column with female-0, male-1
df_train['Gender'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Create Embark column with 'C'- 0, 'Q' - 1, 'S' - 2
df_train.Embarked[df_train.Embarked.isnull(
)] = df_train.Embarked.dropna().mode().values
df_train['Embark'] = df_train['Embarked'].map(
    {'C': 0, 'Q': 1, 'S': 2}).astype(int)
df_test['Embark'] = df_test['Embarked'].map(
    {'C': 0, 'Q': 1, 'S': 2}).astype(int)

# Fill missing value of Age in both train and test dataset
median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df_train[(df_train['Gender'] == i) &
                                     (df_train['Pclass'] == j + 1)]['Age'].dropna().median()

df_train['AgeFill'] = df_train['Age']
df_test['AgeFill'] = df_test['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.Gender == i) & (
            df_train.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (
            df_test.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

# Fill missing value of Fare in test dataset
df_test.Fare[df_test.Fare.isnull()] = df_train.Fare.median()

# Create 'AgeIsNull' column
df_train['AgeIsNull'] = pd.isnull(df_train.Age).astype(int)
df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)

# Feature Engineering
# Family size
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

# Age*Class
df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass

# Collect the test data's PassengerIds before dropping it
test_ids = df_test['PassengerId'].values

# Drop object data
df_train = df_train.drop(
    ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin',
                        'Embarked', 'Age', 'PassengerId'], axis=1)

# Convert to arrays
train_data = df_train.values
test_data = df_test.values


# Modeling
print('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print('Predicting...')
output = forest.predict(test_data).astype(int)

predictions_file = open("../submissions/QuickRFModel-20160514.csv", 'w', newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids, output))
predictions_file.close()
print('Done.')
