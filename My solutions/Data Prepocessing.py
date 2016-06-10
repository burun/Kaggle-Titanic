import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
import csv as csv
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import os
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Pandas DataFrame
df_train = pd.read_csv(os.path.dirname(__file__) + '/../Data Files/train.csv')
df_test = pd.read_csv(os.path.dirname(__file__) + '/../Data Files/test.csv')


# Turn "Sex" column categories to numbers
Sex_le = LabelEncoder()
df_train['Sex'] = Sex_le.fit_transform(df_train['Sex'])
df_test['Sex'] = Sex_le.fit_transform(df_test['Sex'])

# Turn "Embarked" column categories to dummy features (one-hot encoding)
# df_train = pd.concat([df_train, pd.get_dummies(df_train[['Embarked']])],
#                      axis=1)
# df_test = pd.concat([df_test, pd.get_dummies(df_test[['Embarked']])], axis=1)

# Fill missing value of Age in both train and test dataset
median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df_train[(df_train['Sex'] == i) &
                                     (df_train['Pclass'] == j + 1)]['Age'].dropna().median()

df_train['AgeFill'] = df_train['Age']
df_test['AgeFill'] = df_test['Age']


for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.Sex == i) & (
            df_train.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]
        df_test.loc[(df_test.Age.isnull()) & (df_test.Sex == i) & (
            df_test.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

# Fill missing value of Fare in test dataset
df_test.Fare[df_test.Fare.isnull()] = df_train.Fare.median()

# Create "Age*Class" column
df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass

# Create "Family size" column
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

# Extract title information from Name column
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

df_test['Title'] = df_test['Name'].map(lambda x: get_title(x))
df_test['Title'] = df_test.apply(replace_titles, axis=1)

# Collect the test data's PassengerIds before dropping it
test_ids = df_test['PassengerId'].values

# Drop object data
df_train = df_train.drop(
    ['Name', 'Ticket', 'Parch', 'Cabin', 'Embarked', 'Age',
     'PassengerId'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Parch', 'Cabin',
                        'Embarked', 'Age', 'PassengerId'], axis=1)

# From given train dataset split train and test
X = df_train.iloc[:, 1:].values
y = df_train.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.30, random_state=1)

# To calsulate the important features by RandomForestClassifier
# forest = RandomForestClassifier(n_estimators=100)
# forest = forest.fit(X, y)
#
# feature_importance = forest.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
#
# features_list = df_train.columns.values[1::]
# fi_threshold = 9
# important_idx = np.where(feature_importance > fi_threshold)[0]
# important_features = features_list[important_idx]
# # print "\n", important_features.shape[0], "Important features(>", \
# #        fi_threshold, "% of max importance)...\n"#, \
# #        #important_features
# sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
# # get the figure about important features
# pos = np.arange(sorted_idx.shape[0]) + .5
#
# plt.title('Feature Importance')
# plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],
#          color='r', align='center')
# plt.yticks(pos, important_features[sorted_idx[::-1]])
# plt.xlabel('Relative Importance')
# plt.draw()
# plt.show()



# pipe_lr = Pipeline([('scl', StandardScaler()),
#                     ("rf", RandomForestClassifier(n_estimators=100))])
#
# pipe_lr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
#
# test_data = df_test.values
# test_pred = pipe_lr.predict(test_data)
#
# predictions_file = open(os.path.dirname(__file__) + "/../submissions/RFModel-20160610.csv", 'w', newline='')
# open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["PassengerId", "Survived"])
# open_file_object.writerows(zip(test_ids, test_pred))
# predictions_file.close()
