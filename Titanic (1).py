# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn. metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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
    df = df[df[age] <= 75]
    
# create dummy columns:
def getDummies(df, col, dropfirst = True):
    df = pd.get_dummies(data = df, columns = [col], drop_first=dropfirst)
    return df

# COMBINED CLEANING FUNCTIONS:

# cleans all data:
def data_cleaning(csv):     
    df = read_data(csv)
    drop_col(df,'Cabin')
    drop_col(df,'Name')
    drop_col(df,'Ticket')
    drop_col(df,'Embarked')
    #drop_col(df,'Fare')
    df = getDummies(df, 'Sex')
    df = getDummies(df, 'Pclass', False)
    return df
  
# This is just for the TRAIN data. It calls the previous function, and adds any extra steps: 
def train_data_cleaning(csv):
    df = data_cleaning(csv)
    change_dtype(df,'Survived', int)
    change_age(df,'Age')
    return df

# read in and clean TRAIN data:
df = train_data_cleaning('titanic_train.csv')

# Scale TRAIN data
def scale_train_data(X_train, scaler_type):
    scaler = scaler_type
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler

# Scale TEST data or NEW data using existing scaler
def scale_new_data(new_data, scaler):
    new_data_scaled = scaler.fit_transform(new_data)
    return new_data_scaled

# Splits cleaned TRAIN data, scales it(using above functions) and returns all.
def prep_data(df):
    X = df.drop(['Survived', 'PassengerId'], axis = 1) 
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    X_train_scaled, scaler = scale_train_data(X_train, MinMaxScaler())
    X_test_scaled = scale_new_data(X_test, scaler)
    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

#Cleans and scale NEW data:
def clean_scale_data(csv, scaler):
    test_df = data_cleaning(csv)
    test_df.loc[test_df['Age'].isnull(), 'Age']= test_df['Age'].mean()
    test_df.loc[test_df['Fare'].isnull(), 'Fare']= 6.2375
    drop_col(test_df,'PassengerId')
    test_df_scaled = scale_new_data(test_df, scaler)
    return test_df, test_df_scaled



#makes model, output preiction to csv and print scores
def fit_predict_model(model_with_params, model_initial, test_data_scaled, test_data_original):
    model = model_with_params
    model.fit(X_train_scaled, y_train)
    #predit test set of Train data:
    prediction=model.predict(X_test_scaled)
    # predict test data:
    pred= model.predict(test_data_scaled)
    merged=test_data_original.merge(pd.DataFrame(pred), left_on=None, right_on=None, left_index=True, right_index=True)
    output = pd.DataFrame({'PassengerId': merged['PassengerId'], 'Survived':merged[0] })
    #output predictions to csv
    output.to_csv(f'{model_initial}_team8.csv', index=False)
    #get score of model
    cv_scores = cross_val_score(model, X, y, cv=5)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    print(precision, recall)
    print('{} cv_scores mean: {}, Precision: {}, Recall: {}'.format(model_initial, np.mean(cv_scores), precision, recall))

# split and scale our TRAIN data:
X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prep_data(df)


test_path = 'test (1) - test (2).csv'
test_data = pd.read_csv(test_path)
test_df, test_df_scaled = clean_scale_data(test_path, scaler)
test_df_scaled = pd.DataFrame(test_df_scaled, columns = X.columns)

#Call function on all models:

# logistic regression
fit_predict_model(LogisticRegression(max_iter= 500, penalty='l2',C=.5), 'lr', test_df_scaled, test_data)

# Tree
fit_predict_model(DecisionTreeClassifier(criterion='entropy', max_depth=4), 'dt', test_df_scaled, test_data)

#Forest
fit_predict_model(RandomForestClassifier(criterion = 'gini', max_depth = 7, max_features = None, n_estimators = 10), 'rf', test_df_scaled, test_data)

#Knn
fit_predict_model(KNeighborsClassifier(metric='manhattan', n_neighbors=7, weights='distance'), 'kn', test_df_scaled, test_data)