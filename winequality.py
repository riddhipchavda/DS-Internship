#wine quality dataset

import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from pyexpat import model

from scipy.stats import pearsonr, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as CM
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
le = LabelEncoder()
rf = RandomForestClassifier()
lr = LogisticRegression()
nb = MultinomialNB()
dt = DecisionTreeClassifier()


#reading csv
df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/Wine Quality/winequalityN.csv")

print(df.head())

print(df.describe(include='all'))

print(df.isnull().sum())

#filling missing values with mean
print(df['fixed acidity'].fillna((df['fixed acidity'].mean()),inplace=True))
print(df['volatile acidity'].fillna((df['volatile acidity'].mean()),inplace=True))
print(df['citric acid'].fillna((df['citric acid'].mean()),inplace=True))
print(df['residual sugar'].fillna((df['residual sugar'].mean()),inplace=True))
print(df['chlorides'].fillna((df['chlorides'].mean()),inplace=True))
print(df['pH'].fillna((df['pH'].mean()),inplace=True))
print(df['sulphates'].fillna((df['sulphates'].mean()),inplace=True))

#checking unique values in type column
print(df['type'].unique())

#replacing type categories with binary
df['type'] = df['type'].replace(['white','red'], [0,1])

#checking outliers in density with boxplot
sns.boxplot(df['density'])
plt.show()

q1 = df['density'].quantile(0.25)
q3 = df['density'].quantile(0.75)

iqr = q3 - q1
print(iqr)

upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

out1 = np.where(df['density']>=upper)[0]
out2 = np.where(df['density']<=lower)[0]

df.drop(index=out1,inplace=True)
df.drop(index=out2,inplace=True)

#confirming outliers in density with boxplot
sns.boxplot(df['density'])
plt.show()

print(df['quality'].value_counts())

#defining target and input variables
X=df.iloc[:,:12]
y=df.iloc[:,12]

#Random Oversampling
print('Before:')
print(y.value_counts())
x_over,Y_over=ros.fit_resample(X,y)
print('After:')
print(Y_over.value_counts())
y = Y_over
y.value_counts()
X= x_over
X.value_counts()

#dividing into train and test sets and implementing random forest algorithm
x_train, x_test, y_train, y_test= tts(X, y, test_size= 0.3, random_state=1)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

