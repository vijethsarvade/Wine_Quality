import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
wine=pd.read_csv("C://Users//svije//.spyder-py3//Wine_Quality//Data//datasets_4458_8204_winequality-red.csv")

#Objecives:
#1.Build a model to Segregation on Wine into red wine and white wine by using k-means cluster
#2.Build a classification models to predict whether the quality of wine is “good quality” or not
wine.shape
wine.head()
wine.columns
wine.info()
wine.describe()

#1.Data cleaning:
#a.identification of duplicate values
wine.duplicated()
wine.duplicated().sum()
#there are 240 rows which are duplicate
wine.loc[wine.duplicated(),:]
#gives which all are duplicates
wine.loc[wine.duplicated(keep="first"),:]
#keeping the duplicate values first
wine1=wine.drop_duplicates(keep='first')
#drop all the dupicates
wine1.shape

#b.identificatioy missing values
wine1.isnull().sum()
sns.heatmap(wine1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#there are no missing values

#2.Visualization and Exploratory data Analysis
#To classify wine
plt.hist(wine1['fixed acidity'],color='red',bins=50)
sns.set_style('whitegrid')
va=wine1[wine1['Wine colour']=='Red']
sns.distplot(va['fixed acidity'],kde=False,label='Red')
fa=wine1[wine1['Wine colour']=='White']
sns.distplot(fa['fixed acidity'],kde=False,label='White')
plt.legend(prop={'size': 12})

plt.hist(wine1['volatile acidity'],color='red',bins=50)
sns.set_style('whitegrid')
fa=wine1[wine1['Wine colour']=='Red']
sns.distplot(fa['volatile acidity'],kde=False,label='Red')
fa=wine1[wine1['Wine colour']=='White']
sns.distplot(fa['volatile acidity'],kde=False,label='White')
plt.legend(prop={'size': 12})    

plt.hist(wine1['citric acid'],color='red',bins=50)
sns.set_style('whitegrid')
fa=wine1[wine1['Wine colour']=='Red']
sns.distplot(fa['citric acid'],kde=False,label='Red')
fa=wine1[wine1['Wine colour']=='White']
sns.distplot(fa['citric acid'],kde=False,label='White')
plt.legend(prop={'size': 12})   

#Model building
from sklearn.cluster import KMeans
#K-means model with 2 clusters
kmeans = KMeans(n_clusters=2)
#Droping the categorical column
wine2=wine1.drop(['Wine colour'],axis=1)
#Fiting the model
kmeans.fit(wine2)
#Gives centralized values
kmeans.cluster_centers_
#Gives the clusters
kmeans.labels_
#We can also visualize the label
wine2['winelable']=kmeans.labels_
wine2.winelable[wine2['winelable']==1]='Red wine'
wine2.winelable[wine2['winelable']==0]='White wine'
wine3=wine2['winelable']

#Applying Decision tree to the dataset
##1.EDA -We'll use pairplot to check the trend
sns.pairplot(wine1,hue='quality',palette='Set1')

#2.Lets check the correlation 
wine1.corr()
sns.heatmap(wine1.corr(),annot=True,fmt='.1g')
#fixed acidity, citric acid ,free sulfur dioxide, total sulfur dioxide and density is postivily correlated at .70

#3.Using the above infomation, lets visualize and see location, distribution, trend & on what basis each quantity level should be to say Wine is good or bad.  

#since i am checking the wine quality using quality variable 
#lets check distribution of quantity variable
plt.hist(wine1['quality'],color='red',bins=50)
#Most of the data is located between 5 to 7

#fixed acidity
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['fixed acidity'])
#no indication or trend of fixed acidity v/s quality

#volatile acidity
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['volatile acidity'])
#we can see trend is towords down which means volatile acidity should be less if the wine to be good and some outliers are there

#citric acid
sns.barplot(x = 'quality', y = 'citric acid', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['citric acid'])
#we can see trend is towords upwords which means citric acid is more as quality increases and some outliers are there

#residual sugar
sns.barplot(x = 'quality', y = 'residual sugar', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['residual sugar'])
#we can see trend is normal with residual sugar v/s quality and some outliers are there

#chlorides
sns.barplot(x = 'quality', y = 'chlorides', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1.chlorides)
#we can see trend is towords down which means chlorides should be less if the wine to be good and some outliers are there

#free sulfur dioxide
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['free sulfur dioxide'])
#we can see trend is towords down after 5 quantity which means free sulfur dioxide should be more if the wine to be good and some outliers are there

#total sulfur dioxide
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1['total sulfur dioxide'])
#we can see trend is towords down after 5 quantity which means total sulfur dioxide should be more if the wine to be good and some outliers are there

#density
sns.barplot(x = 'quality', y = 'density', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1.density)
#no indication or trend and some outliers are there

#pH
sns.barplot(x = 'quality', y = 'pH', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1.pH)
#no indication or trend and some outliers are there

#sulphates
sns.barplot(x = 'quality', y = 'sulphates', data = wine1)
sns.boxplot(x=wine1.quality,y=wine1.sulphates)
#we can see trend is towords upwords which means sulphates is more as quality increases and some outliers are there

#alcohol
sns.barplot(x = 'quality', y = 'alcohol', data = wine1)
sns.boxplot(x=wine1.quality,y=wine.alcohol)
#we can see trend is towords upwords which means alcohol is more as quality increases and some outliers are there

#4.Standardiz the Features to equalize the range of the data and also keep threshold value = 3 to find the outlier
# for Standizing and rescaling import StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler
wine4=wine1.drop(['quality','Wine colour'],axis=1)
wine4=StandardScaler().fit_transform(wine4)
#For outliers we keep threshold = 3
wine4=np.abs(wine4)
wine5=wine4[(wine4<3).all(axis=1)]
wine5=pd.DataFrame(wine5)
wine5_cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density',
                    'pH','sulphates','alcohol']
wine5.columns=wine5_cols
wine6=pd.DataFrame(wine1.quality)
wine6.reset_index(inplace=True)
wine6=wine6.drop(['index'],axis=1)
wine7=pd.concat([wine5,wine6],axis=1)
wine7=wine7.dropna()
sns.boxplot(x=wine7.quality,y=wine7.alcohol)
sns.boxplot(x=wine7.quality,y=wine7.sulphates)
#thus no outliers and the records are 

#5.Converting the quality values to 0 and 1
#with the above annalysis as most of the trand is towords up words,i can justify that if most of the quantity is more then wine will be good
#For effective classification i need to change the output of quantity to binary form by seting the threshold limit to 7.
#wine as ‘good quality’ if quality score is 7 or higher, and if it had a score of less than 7, it is ‘bad quality’.
wine7['gdquality'] = [1 if x >= 7 else 0 for x in wine7['quality']]
wine7['gdquality'].value_counts()
#thus we have 1("good quality)=653 and 0("bad quality)=586
sns.countplot(wine7['gdquality'])


#Model building 
#1.Assigning the values 
X = wine7.drop(['quality', 'gdquality'],axis=1)
y = wine7['gdquality']

#2.Splitting the data in ration of 70:30 for train and test import train_test_split from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)                

#3.Fitting the model in train data set, to fit the model import DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
#training the classifier 
dtree.fit(X_train,y_train)

#4.Predictions, applying the trained classifier to the test data
predictions = dtree.predict(X_test)
predictions

#5.Accuracy check, misclassification check, import classification_report,confusion_matrix from sklearn
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#5.a.Check for misclassification 
count_misclassified = (y_test != predictions).sum()
print('Misclassified samples: {}'.format(count_misclassified))
#out of 408 test data, 123 is missclassified and 285 is correcty classified
#5.b. Accuracy check import metrics from sklearn
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, predictions)
print('Accuracy: {:.2f}'.format(accuracy))
#classification rate is at accuracy of 70%

#6.Tree Evaluation diagram ok
from io import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
features = list(X.columns[0:])
features
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("wine2.png")
Image(graph.create_png()) 














