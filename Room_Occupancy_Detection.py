# -*- coding: utf-8 -*-
# Auto-converted from Jupyter Notebook (.ipynb)
# Source: Room Occupancy Detection.ipynb

# %% [markdown] (cell 1)
# # Room Occupancy Detection

# %% [markdown] (cell 2)
# The aim of this project is to predict whether a room is occupied or not based on the data collected from the sensors. The data set is collected from the UCI Machine Learning Repository. The data set contains 7 attributes. The attributes are date, temperature, humidity, light, CO2, humidity ratio and occupancy. The data set is divided into 3 data sets for training and testing. The data set provides experimental data used for binary classification (room occupancy of an office room) from Temperature, Humidity, Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.

# %% [markdown] (cell 3)
# ##### Data Dictionary
# 
# 
# 
# 
# 
# | Column   Position 	| Atrribute Name 	| Definition                                                                                           	| Data Type    	| Example                                        	| % Null Ratios 	|
# |-------------------|----------------|---------------------------------------|--------------|------------------------------------------------|---------------|
# | 1                 	| Date           	| Date & time in year-month-day hour:minute:second format                                              	| Qualitative  	| 2/4/2015 17:57, 2/4/2015 17:55, 2/4/2015 18:06		 	| 0             	|
# | 2                 	| Temperature    	| Temperature in degree Celcius                                                                        	| Quantitative 	| 23.150, 23.075, 22.890                         	| 0             	|
# | 3                 	| Humidity       	| Relative humidity in percentage                                                                      	| Quantitative 	| 27.272000, 27.200000, 27.390000                	| 0             	|
# | 4                 	| Light          	| Illuminance measurement in unit Lux                                                                  	| Quantitative 	| 426.0, 419.0, 0.0	                              	| 0             	|
# | 5                 	| CO2            	| CO2 in parts per million (ppm)                                                                       	| Quantitative 	| 489.666667,   495.500000, 534.500000           	| 0             	|
# | 6                 	| HumidityRatio  	| Humadity ratio:  Derived quantity from temperature and   relative humidity, in kgwater-vapor/kg-air  	| Quantitative 	| 0.004986, 0.005088, 0.005203                   	| 0             	|
# | 7                 	| Occupancy      	| Occupied or not: 1 for occupied and 0 for not occupied                                               	| Quantitative 	| 1, 0                                           	| 0             	|

# %% (cell 4)
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] (cell 5)
# Loading two datasets and combining them into one dataset

# %% (cell 6)
#loading the datasets
df1 = pd.read_csv('datatest.csv')
df2 = pd.read_csv('datatraining.csv')

# %% (cell 7)
#combining the datasets
df = pd.concat([df1,df2])
df.head()

# %% [markdown] (cell 8)
# ## Data Preprocessing

# %% (cell 9)
#number of rows and columns
df.shape

# %% (cell 10)
#checking for null values
df.isnull().sum()

# %% (cell 11)
#checking for duplicate values
df.duplicated().sum()

# %% (cell 12)
#removing the duplicate values
df.drop_duplicates(inplace=True)

# %% (cell 13)
#checking data types
df.dtypes

# %% (cell 14)
#converting the date and time to datetime format
df['date'] = pd.to_datetime(df['date'])

# %% (cell 15)
df.dtypes

# %% (cell 16)
#checking the descriptive statistics
df.describe()

# %% (cell 17)
#value counts for the target variable i.e. occupancy
df['Occupancy'].value_counts()

# %% [markdown] (cell 18)
# ## Exploratory Data Analysis

# %% [markdown] (cell 19)
# In the exploratory data analysis, we will be looking at the distribution of the data, along with the time series of the data. We will also be looking at the correlation between the variables.

# %% [markdown] (cell 20)
# #### Visualizing the temperture fluctuations over time

# %% (cell 21)
#lineplot for themperature changes for time
plt.figure(figsize=(20,10))
sns.lineplot(x='date',y='Temperature',data=df)
plt.show()

# %% [markdown] (cell 22)
# The spikes in the graph clearly indicates that the room temperature incresases suddenly which might be due to the presence of people in the room. The temperature of the room may increase due to the heat emitted by the human body.

# %% [markdown] (cell 23)
# #### Visualizing the humidity fluctuations over time

# %% (cell 24)
#lineplot for humidity changes for time
plt.figure(figsize=(20,10))
sns.lineplot(x='date',y='Humidity',data=df)
plt.show()

# %% [markdown] (cell 25)
# The line graph between 3rd of February to 6th of February shows some similarity with the temperature graph, which might be due to the presence of people in the room. However 7th of February onwards there has been a significant rise in the humidity levels which might be due to cleaning of the room, or change in the weather conditions. Out of which room cleaning such sweeping the floor might be the reason for the sudden rise in the humidity levels. But it couldn't explain the increase in the humidity levels near 10th of February.

# %% [markdown] (cell 26)
# #### Visualizing the light fluctuations over time

# %% (cell 27)
#lineplot for light changes for time
plt.figure(figsize=(20,10))
sns.lineplot(x='date',y='Light',data=df)
plt.show()

# %% [markdown] (cell 28)
# If we look closely, we can see that the number of peaks in this graph and in the temperature graph are same. This indicates that lights were turned on when there was a person in the room. This is a good indicator of the occupancy of the room.

# %% [markdown] (cell 29)
# ##### Visualizing the CO2 fluctuations over time

# %% (cell 30)
#lineplot for co2 changes for time
plt.figure(figsize=(20,10))
sns.lineplot(x='date',y='CO2',data=df)
plt.show()

# %% [markdown] (cell 31)
# The co2 graph also shows the spikes in the co2 levels which indicates the presence of person in the room, assuming that there is no other source of co2 in the room.In addition to that the spikes also shows correspondence with the temperature graph and light graph. However from 7th of February to 9th of February, the co2 levels where minimum, which indicstes that the room was not occupied during that time. This observation contradicts with the humidity graph and temperature graph.

# %% [markdown] (cell 32)
# #### Visualizing the humidity ratio fluctuations over time

# %% (cell 33)
#lineplot for humidity ratio changes for time
plt.figure(figsize=(20,10))
sns.lineplot(x='date',y='HumidityRatio',data=df)
plt.show()

# %% [markdown] (cell 34)
# The humidity ratio graph is quite similar to the humidity graph. The spikes in the graph indicates the presence of people in the room. Moreover the same assumption is made about the humidity ratio after 9th of February.

# %% [markdown] (cell 35)
# ## Correlation between the variables

# %% [markdown] (cell 36)
# ### Correlation Heatmap

# %% (cell 37)
#correlation heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

# %% [markdown] (cell 38)
# There is a strong coorelation between light and occupancy as well as between humidity and humidity ratio. The co2 levels and temperature also shows a strong correlation with the occupancy. However the humidity and humidity ratio has very less correlation with the occupancy.

# %% [markdown] (cell 39)
# #### Temperature and Occupancy

# %% (cell 40)
#violinplot for temperature
sns.violinplot(y = df['Temperature'],x = df['Occupancy'])
plt.xlabel('Occupancy')
plt.ylabel('Temperature')
plt.show()

# %% [markdown] (cell 41)
# The temperature and occupancy graph shows that the temperature of the room increases when there is a person in the room. This is because of the heat emitted by the human body. The temperature of the room decreases when there is no person in the room. This proves the hypothesis regarding the peaks in the temperature graph.

# %% [markdown] (cell 42)
# #### Light and Occupancy

# %% (cell 43)
#boxplot for light
sns.boxplot(y = df['Light'],x = df['Occupancy'])
plt.xlabel('Occupancy')
plt.ylabel('Light')
plt.show()

# %% [markdown] (cell 44)
# The light intensity of the room increases when there is a person in the room. This is because the lights are turned on when there is a person in the room. The light intensity of the room decreases when there is no person in the room. This proves the hypothesis regarding the peaks in the light graph. The outliers in the boxplot and the curves in the ligh graph might be due to sunlight entering the room.

# %% [markdown] (cell 45)
# #### CO2 and Occupancy

# %% (cell 46)
#violinlot for co2
sns.violinplot(y = df['CO2'],x = df['Occupancy'])
plt.xlabel('Occupancy')
plt.ylabel('CO2')
plt.show()

# %% [markdown] (cell 47)
# The CO2 levels of the room increases when there is a person in the room. This is because of the CO2 emitted by the human body. The CO2 levels of the room decreases when there is no person in the room. This proves the hypothesis regarding the peaks in the CO2 graph.

# %% [markdown] (cell 48)
# From the above EDA, it is quite clear that the temperature, light and CO2 levels of the room are a good indicator of the occupancy of the room. Therefore we will be using these three variables for our classification model.

# %% [markdown] (cell 49)
# ## Data Preprocessing 2

# %% (cell 50)
#dropping columns humidity, date and humidity ratio
df.drop(['Humidity','date','HumidityRatio'],axis=1,inplace=True)

# %% (cell 51)
df.head(10)

# %% [markdown] (cell 52)
# ## Train Test Split

# %% (cell 53)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop(['Occupancy'],axis=1),df['Occupancy'],test_size=0.2,random_state=42)

# %% [markdown] (cell 54)
# ## Model Building

# %% [markdown] (cell 55)
# ### Random Tree Classifier

# %% (cell 56)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc

# %% [markdown] (cell 57)
# ##### Training the model

# %% (cell 58)
#training the model
rfc.fit(x_train,y_train)
#training accuracy
rfc.score(x_train,y_train)

# %% [markdown] (cell 59)
# ## Model Evaluation

# %% (cell 60)
rfc_pred = rfc.predict(x_test)

# %% (cell 61)
#confusion matrix heatmap
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test,rfc_pred),annot=True)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.show()

# %% (cell 62)
#distribution plot for the predicted and actual values
ax = sns.distplot(y_test,hist=False,label='Actual', color='r')
sns.distplot(rfc_pred,hist=False,label='Predicted',color='b',ax=ax)
plt.show()

# %% (cell 63)
from sklearn.metrics import classification_report
print(classification_report(y_test,rfc_pred))

# %% (cell 64)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# %% (cell 65)
print('Accuracy Score : ' + str(accuracy_score(y_test,rfc_pred)))
print('Precision Score : ' + str(precision_score(y_test,rfc_pred)))
print('Recall Score : ' + str(recall_score(y_test,rfc_pred)))
print('F1 Score : ' + str(f1_score(y_test,rfc_pred)))

# %% [markdown] (cell 66)
# ## Testing the model on new dataset

# %% (cell 67)
df_new = pd.read_csv('datatest2.csv')
df_new.head()

# %% (cell 68)
#dropping columns humidity, date and humidity ratio
df_new.drop(['Humidity','date','HumidityRatio'],axis=1,inplace=True)

# %% (cell 69)
#splitting the target variable
x = df_new.drop(['Occupancy'],axis=1)
y = df_new['Occupancy']

# %% (cell 70)
#predicting the values
pred = rfc.predict(x)

# %% (cell 71)
#confusion matrix heatmap
sns.heatmap(confusion_matrix(y,pred),annot=True)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.show()

# %% (cell 72)
#distribution plot for the predicted and actual values
ax = sns.distplot(y,hist=False,label='Actual', color='r')
sns.distplot(pred,hist=False,label='Predicted',color='b',ax=ax)
plt.show()

# %% (cell 73)
print(classification_report(y,pred))

# %% (cell 74)
print('Accuracy Score : ' + str(accuracy_score(y,pred)))
print('Precision Score : ' + str(precision_score(y,pred)))
print('Recall Score : ' + str(recall_score(y,pred)))
print('F1 Score : ' + str(f1_score(y,pred)))

# %% [markdown] (cell 75)
# ## Conclusion

# %% [markdown] (cell 76)
# From the above models we can see that the Random Forest Classifier has the highest accuracy score of 98%. Therefore we will be using the Random Forest Classifier for our final model.
# I also conclude that from the exploratory data analysis, it was found that the change in room temperature, CO levels and light intensity can be used to predict the occupancy of the room, inplace of humidity and humidity ratio.
