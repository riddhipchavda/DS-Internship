#blackfidaysales dataset

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
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

le = LabelEncoder()
rf = RandomForestClassifier()
lr = LinearRegression()
nb = MultinomialNB()
dt = DecisionTreeClassifier()

#reading csv
df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/Black Friday Sales/train.csv")

print(df.head())

df.info()

print(df.describe())

#dealing with missing values
print(df.isnull().sum())

df['Product_Category_2'].fillna(df['Product_Category_2'].median(),inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].median(),inplace=True)

print(df.isnull().sum())

#dropping unwanted cols
df.drop('User_ID',axis = 1,inplace = True)
df.drop('Product_ID',axis = 1,inplace = True)
print(df.head())

#checking outliers using boxplot
sns.boxplot(df['Purchase'])
plt.show()

Q1 = df['Purchase'].quantile(0.25)
Q3 = df['Purchase'].quantile(0.75)

IQR = Q3 - Q1
print('the IQR value is:',IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print('The UPPER value is:',upper)
print('The LOWER value is:',lower)

df['Purchase'].where(df['Purchase']<upper, df['Purchase'].median(),inplace =True)
df['Purchase'].where(df['Purchase']>lower, df['Purchase'].median(),inplace =True)

print(df['Purchase'])

#confirming outlier removal with boxplot
sns.boxplot(df['Purchase'])
plt.show()

#plotting probability plot
stats.probplot(df.Purchase, plot=plt)
plt.show()

df.Purchase = pd.Series(stats.boxcox(df.Purchase,lmbda=stats.boxcox_normmax(df.Purchase)))
stats.probplot(df.Purchase, plot=plt)
plt.show()

#label encoding
print(df.Gender)
l1 = LabelEncoder()
df['Gender'] = l1.fit_transform(df['Gender'])
print(df['Gender'])                                     #...male = 1 female = 0
print(df.head())

l2 = LabelEncoder()
df['City_Category'] = l2.fit_transform(df['City_Category'])
print(df['City_Category'])
print(df.head())

l3 = LabelEncoder()
df['Age'] = l3.fit_transform(df['Age'])
print(df['Age'])
print(df.head())

l4 = LabelEncoder()
df['Stay_In_Current_City_Years'] = l4.fit_transform(df['Stay_In_Current_City_Years'])
print(df['Stay_In_Current_City_Years'])
print(df.head())

X = df.iloc[:,:9]
y = df.iloc[:,9]

print(X)
print(y)

#dividing data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size = 0.3)

#implementing linear regression
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(mse)

r2 = r2_score(y_test,y_pred)
print(r2)