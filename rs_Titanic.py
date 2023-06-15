# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn import tree



# SINGLE CLEANING FUNCTIONS:

# read in data:
def read_data(filepath):
    df = pd.read_csv(filepath)
    return df

#change a column to a different datatype
def change_dtype(df, col, type):
    df[col] = df[col].astype(type)
    
# drop a column
def drop_col(df,col):
    df.drop(columns=col, axis = 1, inplace = True)
    
# change values of a column using a mapping dictionary
def change_type(df, col, map_dict):
    df[col] = df[col].map(map_dict)
    
# drop rows with nans in a specific column:
def drop_nan(df,col):
    df = df[df[col].notna()]
    
# fill in nan values in the age column, and remove outliers
def change_age(df,age):
    df.loc[(df['Age'].isnull())&(df['Survived'] ==1), 'Age']= 28.34
    df.loc[(df['Age'].isnull())&(df['Survived']==0), 'Age']= 30.62
    df = df[df[age] <= 74]
    
# create dummy columns:
def getDummies(df, col, dropfirst = True):
    df = pd.get_dummies(data = df, columns = [col], drop_first=dropfirst)
    return df

# COMBINED CLEANING FUNCTIONS:

# cleans all data:
def data_cleaning(csv):     
    df = read_data(csv)
    #change_dtype(df,'Survived', int)
    drop_col(df,'Cabin')
    drop_col(df,'PassengerId')
    drop_col(df,'Name')
    drop_col(df,'Ticket')
    drop_col(df,'Embarked')
    #change_type(df,'Embarked',{'S':1,'C':2,'Q':3})
    df = getDummies(df, 'Sex')
    df = getDummies(df, 'Pclass', False)
    return df
  
# This is just for the TRAIN data. It calls the previous function, and adds any extra steps: 
def train_data_cleaning(csv):
    df = data_cleaning(csv)
    change_age(df,'Age')
    #drop_nan(df,'Embarked')
    return df

# read in and clean TRAIN data:
df = train_data_cleaning('titanic_train.csv')

# Scale TRAIN data
def scale_train_data(X_train, scaler_type):
    scaler = scaler_type
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler

def scale_test_data(X_test, scaler_type):
    scaler = scaler_type
    X_test_scaled = scaler.fit_transform(X_test)
    return X_test_scaled, scaler

# Scale TEST data or NEW data using existing scaler
def scale_new_data(new_data, scaler):
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

# Splits cleaned TRAIN data, scales it(using above functions) and returns all.
def prep_data(df):
    #X = df.drop('Survived', axis = 1) 
    #y = df['Survived']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    #X_train_scaled, scaler = scale_train_data(X_train, MinMaxScaler())
    scale_test_data(df,MinMaxScaler())
    X_test_scaled = scale_new_data(df, scaler)
    return  X_test_scaled, scaler

#Cleans and scale NEW data:
def clean_scale_data(csv, scaler):
    test_data = data_cleaning(csv)
    test_data_scaled = scale_new_data(test_data, scaler)
    return test_data, test_data_scaled


df = read_data(R"C:\Users\EzrasTorah\Downloads\test (1) - test (1).csv")

df = data_cleaning(df)

# split and scale our TRAIN data:
X, y, X_train, X_test,  X_train_scaled, X_test_scaled, scaler = prep_data(df)



#setting parameter options:
param_grid = {'n_estimators':range(10,500,50),'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': range(3,10,2), 'max_features': ['sqrt', 'log2', None]}

#creating gridsearch instance:

rf = RandomForestClassifier()
rfgs = GridSearchCV(estimator=rf,
                    param_grid=param_grid,
                    cv=5,
                    verbose=3,
                    n_jobs=-1)

#fitting model:
rfgs.fit(X_train, y_train)

rfbest_est = rfgs.best_estimator_
rfbest_est.fit(X_train_scaled, y_train)
rfbest_est_pred_train = rfbest_est.predict(X_train_scaled)

merged=df.merge(pd.DataFrame(rfbest_est_pred_train), left_on=None, right_on=None, left_index=True, right_index=True)
merged.head()

output = pd.DataFrame({'PassengerId': merged['PassengerId'], 'Survived':merged[0] })
output.to_csv('rf_team8.csv', index=False)
output.head()