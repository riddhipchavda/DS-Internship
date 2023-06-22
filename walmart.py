#walmart dataset

import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from pyexpat import model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

le = LabelEncoder()
linr = LinearRegression()

#reading csv
train_df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/walmart-recruiting-store-sales-forecasting/train.csv")
test_df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/walmart-recruiting-store-sales-forecasting/test.csv")

print(train_df.head())
print(test_df.head())

#checking for null values
print(train_df.isnull().sum())

#label encoding for IsHoliday column
l1 = LabelEncoder()
train_df['IsHoliday'] = l1.fit_transform(train_df['IsHoliday'])
print(train_df['IsHoliday'])

#dropping unwanted cols
train_df.drop(['Date'], axis = 1, inplace=True)
print(train_df.head())

#counting weekly sales
print(train_df['Weekly_Sales'].value_counts())

#boxplot of weekly sales
sns.boxplot(train_df['Weekly_Sales'])
plt.show()

#defining target and input variables
y = train_df['Weekly_Sales']
X = train_df.iloc[:,:]
X.drop(['Weekly_Sales'], axis = 1, inplace = True)

print(y)

#dividing data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size = 0.3)

#implementing linear regression
linr.fit(X_train,y_train)
y_pred = linr.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(r2)
