import pandas as pd
from sklearn.datasets import load_iris

irs = load_iris()

print(irs)
print(irs.keys())
print(irs.data)
print(irs.target)
print(irs.feature_names)
print(irs.target_names)
print(irs.DESCR)

df = pd.read_csv("C:/Users/admin/Desktop/pp11dataset/Iris.csv")

print(df)
print(df.head(12))
print(df.tail(6))
print(df.columns.values)
print(df.describe())

#preparing x and y
x= df.drop('Id', axis=1)
x= x.drop('Species', axis=1)
y= df['Species']  # is taken as a list

print(x)
print(y)

#feature selection is an imp task in feature engg
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Specs','Score']
print(featureScores)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

df['SepalLengthCm']=pd.cut(df['SepalLengthCm'],3,labels=['0','1','2'])
df['SepalWidthCm']=pd.cut(df['SepalWidthCm'],3,labels=['0','1','2'])
df['PetalLengthCm']=pd.cut(df['PetalLengthCm'],3,labels=['0','1','2'])
df['PetalWidthCm']=pd.cut(df['PetalWidthCm'],3,labels=['0','1','2'])

print(df)

x= df.drop('Id', axis=1)
x= x.drop('Species', axis=1)
y= df['Species']
print(y)
le=LabelEncoder()

# #dealing with missing values
# 1. Use Drop (df.drop())
# 2. use replace (df.replace("back","DOS"))
# 3. fill NA ()
# df['Item_Weight'].fillna((df['Item_Weight'].mean()/.median()/.mode()), inplace=True)
# df['outlet_Size'].fillna(('Medium'), inplace=True)

print(df.isnull().sum())
print(df.notnull().sum())
print(df.isnull())

df['PetalWidthCm'].fillna((df['PetalWidthCm'].mean()), inplace=True)
print(df)

print(df.isnull().sum())

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x, y)
print(Counter(y))

print(df)

print(Counter(y))
from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x,y = sms.fit_resample(x,y)
print(Counter(y))

print(Counter(y))
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
x, y = rus.fit_resample(x, y)
print(Counter(y))


#outliers
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['SepalLengthCm'])
plt.show()

#dealing w outliers using iqr

print(df['SepalLengthCm'])
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(upper)
print(lower)

out1=df[df['SepalLengthCm']<lower].values
out2=df[df['SepalLengthCm']>upper].values

df['SepalLengthCm'].replace(out1,lower,inplace=True)
df['SepalLengthCm'].replace(out2,upper,inplace=True)

print(df['SepalLengthCm'])

# training dataset
# 80-20
# 70-30
# 90-10

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)

x= df.drop('Id', axis=1)
x= x.drop('Species', axis=1)
y= df['Species']

pca.fit(x)
x=pca.transform(x)

print(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.3)

logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)
print(accuracy_score(y_test,y_pred))