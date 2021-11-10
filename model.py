# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


# preprocessing
train_path = "/home/linda/Desktop/Titanic/titanic/train.csv"
test_path = "/home/linda/Desktop/Titanic/titanic/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train.head(5))
print(test.head(5))

## info
print(train.info())

## data missing
missed_val = (train.isna().sum()/len(train)*100).sort_values(ascending=False)
print(missed_val)

##unique values
unique_val = train.nunique().sort_values()
print(unique_val)

'''Missing values

Case 1: 'Cabin' 77% of missing values. As long as there is 3/4 of the data missing if we
         would decide to mock the data it would not be trustable as long as we are are 
         setting it by ourselves, so the most fair way to proceed is to drop this one.

Case 2: 'Age' with 20% of missing values. With a 20% of missing values we should try to 
        fill following some strategy in order to apply the filling closer to what would be

Case 3: 'Embarked' with 0.2% of missing values. Less than a 0.5% of missing values let us 
         to take a different strategy as long as filling the missing values would affect 
         nearly nothing to results. So in this case we will drop the cases where this property
         is not present
    
   Categorical Values

Case 1: 'Sex' as long as it only has 2 possibles values we can do it manually or by a label encoder.

Case 2: 'Name' This property doesn't give useful info so drop is the best option.

Case 3: 'Ticket' This property doesn't give useful info. Dtrop is the best option too.

Case 4: 'Cabin' drop by missing 70% of values, also not very useful info at first sight. Maybe with less 
         missing could be useful as "travellers on stern side of the boat survived more than travellers 
         on bow side", but 77.1% is too much missing.

Case 5: 'Embarked' has 3 possible values. I could use one-hot but for now I feel more confident doing by 
         hand (considering this is my first attemp on Kaggle).'''

print(train['Embarked'].value_counts())

# Data cleaning 
def CleaningData(data):

    #data missing and categorical to drop
    data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

    #data missing case2
    data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x : x.fillna(x.median()))

    # Fare data missing in test
    data['Fare'] = data.groupby(['Pclass', 'Sex'])['Fare'].transform(lambda x : x.fillna(x.median()))

    #data missing case3
    data.dropna(axis=0, subset=['Embarked'], inplace=True)

    #categorical data
    encoder = preprocessing.LabelEncoder()

    #Embarked
    data['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)
    return data

clean_train = CleaningData(train)
clean_test = CleaningData(test)

## reviewing data cleaning
print(clean_train.info())
print(clean_test.info())

# model
X = pd.get_dummies(train.drop('Survived', axis=1))
y = train['Survived']

## split into test train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def fitAndPredict(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)

## training model
model1 = DecisionTreeClassifier()
model2 = GNB()
model3 = SVC()
model4 = RandomForestClassifier()
model5 = GradientBoostingClassifier()
model6 = KNeighborsClassifier()
model6 = LogisticRegression(solver='liblinear', random_state=42)
model7 = SGDClassifier()

## printing accuracy
models = [model1,model2, model3, model4, model5, model6, model7]
i = 0
for model in models:
    i +=1
    print("Model ", i, ":", model)
    print("Accuracy: ", fitAndPredict(model))

# Since GradientBoosting Classifier is best, let's do some tune it!
tune_model = GradientBoostingClassifier(min_samples_split=20, min_samples_leaf=60, max_depth=3, max_features=7)
print("Final accuracy: ", fitAndPredict(tune_model))

# Let's save predicted results
predict = tune_model.predict(pd.get_dummies(clean_test))

output = pd.DataFrame({'PassengerId': clean_test.PassengerId, 'Survived': predict})
output.to_csv('my_submission.csv', index=False)
print("Submission saved")