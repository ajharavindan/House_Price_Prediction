# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:05:50 2020

@author: Haravindan
"""
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)

#Creating a dataframe using pandas
dataset = pd.DataFrame(boston.data)
dataset.columns = boston.feature_names
dataset['PRICE'] = boston.target

#Spliting the dataset into 2 parts
X = dataset.iloc[:,0:13]    #Creating Features of Independent variable Matrix 
y = dataset.iloc[:,13:]     #Creating Dependent variable vector

#Splitting the dataset into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# #Feature Scaling (In case needed)
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Fitting Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

#Predicting Test set Result
y_pred = regressor.predict(X_test)

#Visualising the Predicted and the Actual values on Test set
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices VS Predicted Prices")

#Feature Selection
#Building an optimal model using Backward Stepwise Regression(Backward Elimination with significance level = 0.05)
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((506,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Predicting Test set Result on Optimal Model
X_test_opt = X_test
X_test_opt = X_test_opt.drop("AGE", axis = 1)   #Droping the coloumns on Test Set which made the "P value" greater the significance level
X_test_opt = X_test_opt.drop("INDUS", axis = 1)
X_test_opt = np.append(arr = np.ones((102,1)).astype(int), values = X_test_opt, axis = 1)
y_pred_opt = regressor_OLS.predict(X_test_opt)

#Visualising the Predicted and the Actual values on Test set with Optimal Model
plt.scatter(y_test,y_pred_opt)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices VS Predicted Prices")

# Saving the model for future prediction(Maybe using a front end application through flask framework)
import pickle
with open('rf_regressor.pickle', 'wb') as file:
	pickle.dump(regressor_OLS, file, pickle.HIGHEST_PROTOCOL)





