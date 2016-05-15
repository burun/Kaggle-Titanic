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

feature_importance = forest.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

features_list = df_train.columns.values[1::]
fi_threshold = 9
important_idx = np.where(feature_importance > fi_threshold)[0]
important_features = features_list[important_idx]
# print "\n", important_features.shape[0], "Important features(>", \
#        fi_threshold, "% of max importance)...\n"#, \
#        #important_features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
#get the figure about important features
pos = np.arange(sorted_idx.shape[0]) + .5

plt.title('Feature Importance')
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], \
       color='r',align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.draw()
plt.show()
