#boston housing dataset

import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from pyexpat import model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn import preprocessing

le = LabelEncoder()
rf = RandomForestClassifier()
lr = LogisticRegression()
nb = MultinomialNB()
dt = DecisionTreeClassifier()

#reading csv
df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/HousingData.csv")

#print(df.describe())

#defining x and y based on target variable
x= df.drop('MEDV', axis=1)
y= df['MEDV']  # is taken as a list

#checking for null values
# print(df.isnull().sum())
# print(df.notnull().sum())
#print(df.isnull())

#replacing null values with mean
df['CRIM'].fillna((df['CRIM'].mean()), inplace=True)
#print(df.isnull().sum())

df['ZN'].fillna((df['ZN'].mean()), inplace=True)
#print(df.isnull().sum())

df['INDUS'].fillna((df['INDUS'].mean()), inplace=True)
#print(df.isnull().sum())

df['CHAS'].fillna((df['CHAS'].mean()), inplace=True)
#print(df.isnull().sum())

df['AGE'].fillna((df['AGE'].mean()), inplace=True)
#print(df.isnull().sum())

df['LSTAT'].fillna((df['LSTAT'].mean()), inplace=True)
#print(df.isnull().sum())

#checking for outliers using boxplot
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

#checking outlier percentages
for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    #print("Column %s outliers = %.2f%%" % (k, perc))

df = df[~(df['MEDV'] >= 50.0)]
#print(np.shape(df))

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.histplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#plt.show()

plt.figure(figsize=(20, 10))
sns.heatmap(df.corr().abs(),  annot=True)
# plt.show()


# Let's scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = df.loc[:,column_sels]
y = df['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(10, 5))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#plt.show()

#removing skewness of data using log transform
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])

#linear ridge regression
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

l_regression = linear_model.LinearRegression()
kf = KFold(n_splits=10)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores = cross_val_score(l_regression, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

scores_map = {}
scores_map['LinearRegression'] = scores
l_ridge = linear_model.Ridge()
scores = cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['Ridge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Lets try polinomial regression with L2 with degree for the best fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#for degree in range(2, 6):
#    model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
#    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['PolyRidge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
