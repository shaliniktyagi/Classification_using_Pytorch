#!/usr/bin/env python
# coding: utf-8

# # Python Test
# You have been provided with learning activity data collected from a global sample of students. The attributes are broken down into demographic features like gender and nationality, academic backgrounds like educational stage and grade level, and behavioral observations like engagement counts and parent satisfaction. 
# 
# The students are classified into three ability levels based on how they perform on selected material: 
# 
# __Low__: grades ranging from 0-39 <br>
# __Medium__: grades ranging from 40-89 <br>
# __High__: grades ranging from 90-100 <br>
# 
# You'll have to dive into the data, learn the attributes, and construct a model that can predict the performance level of a given student.  Some common libraries have been loaded, but use any more that you wish.

# # Load the Data
# Load the data into a pandas dataframe. Display the shape of the data, and count the number of null values in each column.

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Load the dataset
data = pd.read_csv('data.csv')
data.head()


# In[3]:


# shape of the dataset
print(data.shape)

# count the number of missing values (null) in each column
print(data.isna().sum())


# # Perform basic EDA
# Use a library of your choice to visualise attributes in the data. Answer each question below with a visualisation. The attribute glossary has also been provided for your reference.
# 
# ## Attribute Glossary
# Gender- student's gender (nominal: 'Male' or 'Female’)
# 
# Nationality- student's nationality (nominal:' Luxembourg',' Lebanon',' Egypt',' USA',' Nigeria',' Venezuela',' Rwanda',' Germany',' France',' India',' Cambodia',' Chile',' Spain')
# 
# Place of birth- student's Place of birth (nominal:' Luxembourg',' Lebanon',' Egypt',' USA',' Nigeria',' Venezuela',' Rwanda',' Germany',' France',' India',' Cambodia',' Chile',' Spain')
# 
# Educational Stages- educational level student belongs (nominal:‘lowerlevel’,’MiddleSchool’,’HighSchool’)
# 
# Grade Levels- grade to which the student belongs (nominal:‘G-01’, ‘G-02’, ‘G-03’, ‘G-04’, ‘G-05’, ‘G-06’, ‘G-07’, ‘G-08’, ‘G-09’, ‘G-10’, ‘G-11’, ‘G-12‘)
# 
# Section ID- classroom student belongs (nominal:’A’,’B’,’C’)
# 
# Topic- course topic (nominal:'Mathematics",'English','Science','Geography','History','French','Spanish','Sport Science’)
# 
# Semester- school year semester (nominal:’First’,’Second’)
# 
# Parent responsible for student (nominal:’mother’,’father’)
# 
# Raised hand- how many times the student raises his/her hand in the classroom (numeric, discrete)
# 
# Used resources- how many times the student uses course content (numeric, discrete)
# 
# Viewed notifications- how many times the student checks their new notifications (numeric, discrete)
# 
# Discussion groups- how many times the student participate on discussion groups (numeric, discrete)
# 
# Parent Answered Survey- parent answered the surveys which are provided from school or not (nominal:’Yes’,’No’)
# 
# Parent School Satisfaction- the degree of parental satisfaction towards the school (nominal:’Yes’,’No’)
# 
# Student Absence Days- the number of absence days for each student (nominal:above-7, under-7)

# ## Show the relation of each column to students' ability level
# 
# Somehow indicate the columns with a positive correlation or relationship to 'abilityLevel'. This could be via a table or a visualisation.

# In[4]:


# fill the missing values with 0
data1 = data.fillna(0) 

# Delete parentSchoolSatisfaction column from the dataframe
data1[['gradeG','gradeLevel']] = data1.gradeLevel.str.split("-",expand=True,)
data2 = data1.drop(['parentSchoolSatisfaction','studentAbsenceDays','gradeG', 'parent'], axis = 1) 

data2.info()


# In[5]:


sns.countplot(x = "abilityLevel", data = data2, palette = "Greens");
plt.show()


# In[7]:


from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
data2["abilityLevel_code"] = ord_enc.fit_transform(data2[["abilityLevel"]])
data2["gender_code"] = ord_enc.fit_transform(data2[["gender"]])
data2["nationality_code"] = ord_enc.fit_transform(data2[["nationality"]])
data2["placeOfBirth_code"] = ord_enc.fit_transform(data2[["placeOfBirth"]])
data2["educationalStage_code"] = ord_enc.fit_transform(data2[["educationalStage"]])
data2["topic_code"] = ord_enc.fit_transform(data2[["topic"]])
data2["sectionId_code"] = ord_enc.fit_transform(data2[["sectionId"]])
data2["semester_code"] = ord_enc.fit_transform(data2[["semester"]])
data2["parentAnsweredSurvey_code"] = ord_enc.fit_transform(data2[["parentAnsweredSurvey"]])

#data2["gradeLevel_code"] = ord_enc.fit_transform(data2[["gradeLevel"]])
#data2["parent_code"] = ord_enc.fit_transform(data2[["parent"]])
data2.head(15)


# In[8]:


data3 = data2.drop(['abilityLevel','gender','nationality', 'placeOfBirth', 'educationalStage', 'topic', 'sectionId', 'semester', 'parentAnsweredSurvey','placeOfBirth_code'], axis = 1)
data3.head(15)


# In[9]:


data3.describe()


# In[10]:


plt.figure(figsize=(20,10))
plt.title('correlation matrix')
sns.heatmap(data3.corr(), cmap = 'RdYlGn', annot = True )
#correlation_mat = data3.corr()

#sns.heatmap(correlation_mat, annot = True)

plt.show()


# In[11]:


# class distribution
print(data3.groupby('abilityLevel_code').size())
data4 = data3.dropna()
data4.isna().sum()


# In[164]:




# example of oversampling a multi-class classification dataset
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
# define the dataset location
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
# load the csv file as a data frame
df = read_csv(data, header=None)
data = df.values


# In[ ]:





# # Model the data

# Answer the questions below with whatever machine learning libraries you choose. The goal here is to predict the 'abilityLevel' attribute for every student.

# ## Prepare the data
# Transform the data in any way you choose. Be ready to explain your reasoning for selecting the columns that need to be transformed as well as the transformations applied. When all transformations are applied, split the data as necessary for prediction.

# In[174]:


import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import torchvision
import torch.utils.data
import torchvision.transforms as transforms

#from keras.utils import to_categorical
import torch.nn.functional as F


# In[209]:


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim,50)
        self.layer2 = nn.Linear(50, 90)
        self.layer3 = nn.Linear(90, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) # To check with the loss function
        return x


# In[210]:


from sklearn.model_selection import train_test_split


# split the dataset into training and test data
data4.astype('float').dtypes
labels = data4['abilityLevel_code']
#labels.to_numpy()
# Remove the labels from the features
features= data4.drop('abilityLevel_code', axis = 1)
features.to_numpy()
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 42,shuffle = True)


print( train_features.shape, train_labels.shape)
print (test_features.shape, test_labels.shape)
#test_features.head()
#print(test_features.head())
#print(test_labels.head())


# In[217]:


# Training
model = Model(train_features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

def print_(loss):
    print ("The loss calculated: ", loss)


# In[218]:


#train_features_np = train_features.to_numpy()
#train_lables_np = train_labels.to_numpy()


torch_tensor_output = torch.tensor(data4['abilityLevel_code'].values)
data_n= features.to_numpy()

grade_f = data_n.astype('float')

torch_tensor_vectors = torch.from_numpy(grade_f)

#train_features, test_features, train_labels, test_labels = train_test_split(torch_tensor_vectors, torch_tensor_output, test_size = 0.30, random_state = 42,shuffle = True)


print( train_features.shape, train_labels.shape)
print (test_features.shape, test_labels.shape)




#grade_np = features['gradeLevel'].to_numpy()

#grade_f = grade_np.astype('float')

#torch_tensor_vectors = torch.from_numpy(grade_f)


# In[219]:


train_features_N = train_features.to_numpy()
train_features_T = train_features_N.astype('float')
#train_vectors = torch.from_numpy(train_features_T)


test_features_N = test_features.to_numpy()
test_features_T = test_features_N.astype('float')
#test_vectors = torch.from_numpy(test_features_T)

train_labels_N = train_labels.to_numpy()
train_labels_T = train_labels_N.astype('float')
#train_vectors = torch.from_numpy(train_labels_T)


test_labels_N = test_labels.to_numpy()
test_labels_T = test_labels_N.astype('float')
#test_vectors = torch.from_numpy(test_labels_T)


# In[220]:


# Not using dataloader
x_train, y_train = Variable(torch.from_numpy(train_features_T)).float(), Variable(torch.from_numpy(train_labels_T)).long()
for epoch in range(1, epochs+1):
    print ("Epoch #",epoch)
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    print_(loss.item())
    
    # Zero gradients
    optimizer.zero_grad()
    loss.backward() # Gradients
    optimizer.step() # Update


# In[221]:


# Prediction
x_test = Variable(torch.from_numpy(test_features_T)).float()
pred = model(x_test)
pred = pred.detach().numpy()


# In[222]:


print ("The accuracy is", accuracy_score(test_labels_T, np.argmax(pred, axis=1)))


# ## Predict student level
# Feed your input data into a model of your choice and observe how it performs. Be ready to explain why you selected this model for your experiment and the metric you used to evaluate performance.

# ## Tune your hyper-parameters
# Demonstrate a tuning of hyper-parameters that will improve performance. You can do this via manual testing or a programatic package. Be ready to explain why those parameters affected model performance.

# In[462]:


#from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
# define dataset
#X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# define models and parameters
model_BC = BaggingClassifier()
n_estimators = [10, 100]
# define grid search
grid = dict(n_estimators=n_estimators)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model_BC, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(train_features,train_labels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[277]:


best_model =grid_result.best_estimator_

predictions = best_model.predict(test_features)


# In[278]:


from sklearn.metrics import mean_squared_error as MSE
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
# Evaluate the test set RMSE
rmse_test = MSE(test_labels, predictions)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# # Extra thoughts for discussion
# 
# What other models might you have chosen?<br>
# What more information might be relevant to classify students?<br>
# If an accurate model is developed, how might it be used to improve a student's experience using CENTURY?
