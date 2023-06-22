
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
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
le = LabelEncoder()
lr = LinearRegression()
cv = CountVectorizer()

#reading json
df = pd.read_json("C:/Users/admin/Desktop/pp11dataset/whats-cooking/cookingtrain.json")

print(df.head())

#dealing with missing values
print(df.isnull().sum())

#checking unique values
print(df.cuisine.nunique())
print(df.cuisine.unique())

#checking value counts
print(df['cuisine'].value_counts())

#mapping cuisines
df['cuisine'] = df['cuisine'].map({'italian':1,
                       'mexican':2,
                       'southern_us':3,
                       'indian':4,
                       'chinese':5,
                       'french':6,
                       'cajun_creole':7,
                       'thai':8,
                       'japanese':9,
                       'greek':10,
                       'spanish':11,
                       'korean':12,
                       'vietnamese':13,
                       'moroccan':14,
                       'british':15,
                       'filipino':16,
                       'irish':17,
                       'jamaican':18,
                       'russian':19,
                       'brazilian':20
})

#defining target and input
X = df.iloc[:,-1]
y = df['cuisine']
print(X)
print(y)

#dividing data in train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

print(X_train)
print(X_test)

X_train['ingredients'] = X_train['ingredients'].apply(lambda x:  ' '.join(x))
X_test['ingredients'] = X_test['ingredients'].apply(lambda x:  ' '.join(x))

print(X_test['ingredients'])

#applying BoW
X_train_bow = cv.fit_transform(X_train['ingredients']).toarray()
X_test_bow = cv.transform(X_test['ingredients']).toarray()

#applying gaussian naive bayes
gnb.fit(X_train_bow,y_train)

y_pred = gnb.predict(X_test_bow)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))