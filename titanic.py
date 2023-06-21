#titanic dataset

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

le = LabelEncoder()
rf = RandomForestClassifier()
lr = LogisticRegression()
nb = MultinomialNB()
dt = DecisionTreeClassifier()

#reading csv
df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/Titanic.csv")

#dropping unwanted cols
cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
df = df.drop(cols, axis=1)

#replacing missing values
df['Embarked'].fillna((df['Embarked'].mode()), inplace=True)
print(df.isnull().sum())

df['Age'].fillna((df['Age'].mean()), inplace=True)
print(df.isnull().sum())

#label encoding to change sex from male female to integers
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])

le.fit(df['Embarked'])
df['Embarked']=le.transform(df['Embarked'])

#defining x and y
x= df.drop('Survived', axis=1)
y= df['Survived']  # is taken as a list

#checking for outliers using boxplot
sns.boxplot(df['Fare'])
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1
print('the IQR value is:',IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print('The UPPER value is:',upper)
print('The LOWER value is:',lower)

out1=df[df['Fare']<lower].values
out2=df[df['Fare']>upper].values

df['Fare'].replace(out1,lower,inplace=True)
df['Fare'].replace(out2,upper,inplace=True)

print(df['Fare'])

#confirming outlier removal using boxplot
sns.boxplot(df['Fare'])
plt.show()

#applying logistic regression
pca=PCA(n_components=7)
pca.fit(x)
x=pca.transform(x)
print(x)

#Dividing data set into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(accuracy_score(y_test,y_pred)) #Accuracy of LR

#-----------------------------------------------END OF LOGISTIC REGESSION----------------------------------------------
#_________________________________________________RANDOM FOREST CLASSSIFICATION________________________________________

train_data = pd.read_csv('C:/Users/admin/Desktop/pp11dataset/traintitanic.csv')
test_data = pd.read_csv('C:/Users/admin/Desktop/pp11dataset/testtitanic.csv')
print(train_data.head())
print(test_data.head())
print(train_data.describe(include='all'))

#Processing missing data and duplicates
print('\nNull Values in Training \n{}'.format(train_data.isnull().sum()))
print('\nNull Values in Testing \n{}'.format(test_data.isnull().sum()))

print('\nDuplicated values in train {}'.format(train_data.duplicated().sum()))
print('Duplicated values in test {}'.format(test_data.duplicated().sum()))

#Filling Embarked and Fare
print('Embarkation per ports \n{}'.format(train_data['Embarked'].value_counts()))

#since the most common port is Southampton the chances are that the missing one is from there
train_data['Embarked'].fillna(value='S', inplace=True)
test_data['Fare'].fillna(value=test_data.Fare.mean(), inplace=True)

print('Embarkation per ports after filling \n{}'.format(train_data['Embarked'].value_counts()))

#Filling Age
mean_age_miss = train_data[train_data["Name"].str.contains('Miss.', na=False)]['Age'].mean().round()
mean_age_mrs = train_data[train_data["Name"].str.contains('Mrs.', na=False)]['Age'].mean().round()
mean_age_mr = train_data[train_data["Name"].str.contains('Mr.', na=False)]['Age'].mean().round()
mean_age_master = train_data[train_data["Name"].str.contains('Master.', na=False)]['Age'].mean().round()

print('Mean age of Miss. title {}'.format(mean_age_miss))
print('Mean age of Mrs. title {}'.format(mean_age_mrs))
print('Mean age of Mr. title {}'.format(mean_age_mr))
print('Mean age of Master. title {}'.format(mean_age_master))
def fill_age(name_age):

        name = name_age[0]
        age = name_age[1]

        if pd.isnull(age):
            if 'Mr.' in name:
                return mean_age_mr
            if 'Mrs.' in name:
                return mean_age_mrs
            if 'Miss.' in name:
                return mean_age_miss
            if 'Master.' in name:
                return mean_age_master
            if 'Dr.' in name:
                return mean_age_master
            if 'Ms.' in name:
                return mean_age_miss
        else:
            return age

train_data['Age'] = train_data[['Name', 'Age']].apply(fill_age, axis=1)
test_data['Age'] = test_data[['Name', 'Age']].apply(fill_age, axis=1)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,5))
sns.heatmap(train_data.isnull(),cmap='copper', ax=ax1)
sns.heatmap(train_data.isnull(), cmap='copper', ax=ax2)
plt.tight_layout()
plt.show()

train_data['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in train_data['Cabin']])
test_data['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in test_data['Cabin']])

plt.figure(figsize=(12,5))
plt.title('Box Plot of Temperatures by Modules')
sns.boxplot(x='Cabin',y='Fare',data=train_data, palette='Set2')
plt.tight_layout()
plt.show()

print('Mean Fare of Cabin B {}'.format(train_data[train_data['Cabin'] == 'B']['Fare'].mean()))
print('Mean Fare of Cabin C {}'.format(train_data[train_data['Cabin'] == 'C']['Fare'].mean()))
print('Mean Fare of Cabin D {}'.format(train_data[train_data['Cabin'] == 'D']['Fare'].mean()))
print('Mean Fare of Cabin E {}'.format(train_data[train_data['Cabin'] == 'E']['Fare'].mean()))

def reasign_cabin(cabin_fare):
    cabin = cabin_fare[0]
    fare = cabin_fare[1]

    if cabin == 'X':
        if (fare >= 113.5):
            return 'B'
        if ((fare < 113.5) and (fare > 100)):
            return 'C'
        if ((fare < 100) and (fare > 57)):
            return 'D'
        if ((fare < 57) and (fare > 46)):
            return 'D'
        else:
            return 'X'
    else:
        return cabin

train_data['Cabin'] = train_data[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)
test_data['Cabin'] = test_data[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)

plt.figure(figsize=(12,5))
plt.title('Box Plot of Temperatures by Modules')
sns.boxplot(x='Cabin',y='Fare',data=train_data, palette='Set2')
plt.tight_layout()
plt.show()

#Feature Engineering

fig, axx = plt.subplots(1, 3, figsize=(20,5))
axx[0].set_title('Amounth of Siblins/Spouses')
sns.countplot(x='SibSp', data=train_data, ax=axx[0])
axx[1].set_title('Amounth of parents/children')
sns.countplot(x='Parch', data=train_data, ax=axx[1])
axx[2].set_title('Distribution of Classes')
sns.countplot(x='Pclass', data=train_data, ax=axx[2])
plt.tight_layout()
plt.show()

print(train_data.isnull().sum())
print(test_data.isnull().sum())

#dropping columns
train_data = train_data.drop(['Name','Ticket','PassengerId'], axis=1)
test_data = test_data.drop(['Name','Ticket','PassengerId'], axis=1)
print(train_data.head())

#labelling categories
categories = {"female": 1, "male": 0}
train_data['Sex']= train_data['Sex'].map(categories)
test_data['Sex']= test_data['Sex'].map(categories)

categories = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked']= train_data['Embarked'].map(categories)
test_data['Embarked']= test_data['Embarked'].map(categories)

categories = train_data.Cabin.unique()
train_data['Cabin'] = train_data.Cabin.astype("category").cat.codes
test_data['Cabin'] = test_data.Cabin.astype("category").cat.codes

plt.figure(figsize=(14,8))
sns.heatmap(train_data.corr(), annot=True)
plt.tight_layout()
plt.show()

#Normalize the data

#Dropping label
LABEL = 'Survived'
y = train_data[LABEL]
train_data = train_data.drop(LABEL, axis=1) #Dropping label to normalize

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

scaled_train = pd.DataFrame(scaled_train, columns=train_data.columns, index=train_data.index)
scaled_test = pd.DataFrame(scaled_test, columns=test_data.columns, index=test_data.index)
print(scaled_train.head())

#Classification

X_train, X_test, y_train, y_test = train_test_split(scaled_train, y, test_size=0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
print(clf.fit(X_train, y_train))
feature_imp = pd.Series(clf.feature_importances_, index=scaled_train.columns).sort_values(ascending=False)
#print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index)

#Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.tight_layout()
plt.show()

#Removing less important features
new_train = scaled_train.drop(['Parch','Embarked'], axis=1)
new_test = scaled_test.drop(['Parch','Embarked'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_train, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(" Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
prediction = clf.predict(new_test)

test_data['Survival_Predictions'] = pd.Series(prediction)
test_data.head()

#____________________________________________________END OF RANDOM FOREST_______________________________________________
#____________________________________________________DECISION TREE______________________________________________________

#reading csv
data = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/Titanic.csv")
print(data.head())

#Using drop function to exclude collums I won't need. The parameter inplace true will change the table
data.drop(["PassengerId", "Name","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], axis="columns",inplace=True)
print(data.head())

#To define X I can simply drop the column I don't need, which is the Y
X = data.drop(["Survived"],axis="columns")
#For the variable Y I can simply select the column Survived
Y = data["Survived"] #Another way to declare: Y = data.Survived
print(Y)

#Pattern: Male = 0 / Female = 1
X.Sex = X.Sex.map({"male":0,"female":1})
print(X.head())

print(X.Age[0:10])

#To fix this, we will use method fillna(). Inside the brackets I'll put the information to replace, which will be the mean
X.Age = X.Age.fillna(X.Age.mean())
print(X.Age[0:10])
#Index 5 was empty and is now showing the mean (29.699118)

#Parameters: X, Y, percentage. If I put one, the method will automatically calculate the other
train_test_split(X,Y,train_size = 0.8) # It is automatically setting 0.2 for testing

# Now I create 4 variables to place each parameter
X_train,X_test,Y_train,Y_real=train_test_split(X,Y,train_size = 0.8)
#to see the length:
print(len(X_train))

#Note that X_test is not in order. It presents it randomly to prevent model from becoming biased
print(X_test)

#data analytics
#Decision Tree
#If it is empty it will create it with as many levels it can. TO avoid this we put max_dep to determine how many levels we want
#The next parameter will be the criteria. If I dont put it will use gini. If I wanted to use entropy I'd have to put it. I want to use gini, so I don't need to put it
model_dt = tree.DecisionTreeClassifier(max_depth=3)   #model with deph 3
print(model_dt)
#Now the method fit will adjust data, training the model
model_dt.fit(X_train,Y_train)

#Using predict method to test the model
model_dt.predict(X_test) #always gets x and retuns y
print(model_dt)

#Storing the result in a variable
Y_pred_dt = model_dt.predict(X_test)
print(Y_pred_dt)

#Dataframe to show Y real and Y predicted:
#Using a dictionary to separate columns
result = pd.DataFrame({"Survided REAL": Y_real, "Survived PREDICTION": Y_pred_dt})
print(result)

#Plotting the tree
plt.figure(figsize=(40,40))
graph_tree = plot_tree(model_dt,feature_names = ['Pclass','Sex','Age'],
                        class_names = ['Survived','Not Survived'],
                        filled=True,rounded=True,fontsize = 8)
plt.show()

#Decision Tree - Performance Measurement
#Accuracy = true negatives + true positives / true positives + false positives + true negatives + false negatives
accuracy_dt = model_dt.score(X_test,Y_real)
print(accuracy_dt)

#Accuracy = true negatives + true positives / true positives + false positives + true negatives + false negatives
accuracy_dt = model_dt.score(X_test,Y_real)
print(accuracy_dt)

#Precision = true positive / true positive + false positive
precision_dt = metrics.precision_score(Y_real,Y_pred_dt)
print(precision_dt)

#_________________________________________________END OF DT___________________________________________________________#
