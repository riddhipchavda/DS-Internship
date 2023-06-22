
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from pyexpat import model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

lor = LogisticRegression()
scaler = StandardScaler()
gnb = GaussianNB()
le = LabelEncoder()
lr = LinearRegression()
cv = CountVectorizer()

#reading csv
df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/adult.csv")

#checking for missing values
#print(df.isnull().sum())

#no missing values

#print(df.head())

#replacing "?" with mode
df.workclass = df.workclass.replace(['?'],df.workclass.mode())
df.occupation = df.occupation.replace(['?'],df.occupation.mode())
df['native.country'] = df['native.country'].replace(['?'],df['native.country'].mode())

#print(df.head())

#defining x and y
X = df.drop(['income'],axis=1)
y = df['income']

#Dividing data set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#label encoding
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
    le = preprocessing.LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)

#implementing logistic regression
lor.fit(X_train, y_train)
y_pred = lor.predict(X_test)

print(accuracy_score(y_test, y_pred))