# Using scikit-learn to implement Simple Linear Regression and Multiple Linear Regression
# Creating a model, training and testing it and use the model

# Understanding the Data

# Fuel_Consumption_Ratings.csv:
# I have downloaded a fuel consumption data, Fuel_Consumption_Ratings.csv, which contains
# model-specific fuel consumption ratings and estimated carbon dioxide emissions
# for new light-duty vehicles for retail sale in Canada.
#
# Variables:
# Model Year e.g. 2014
# Make e.g. Acura
# Model e.g. ILX
# Vehicle Class e.g. SUV
# Engine Size e.g. 4.7
# Cylinders e.g 6
# Transmission e.g. A6
# Fuel e.g. Z
# Fuel Consumption City (L/100 km) e.g. 9.9
# Fuel Consumption Hwy (L/100 km) e.g. 8.9
# Fuel Consumption (L/100 km) e.g. 9.2
# Fuel Consumption (mpg) e.g. 33
# CO2 Emissions (g/km) e.g. 182 --> low --> 0
# CO2 Rating e.g. 6
# Smog Rating e.g. 3

# Solution 1: Simple Linear Regression with OLS Using Scikit-Learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Read the data

df = pd.read_csv(r"C:\Users\güneş market\Desktop\exceld\Fuel_Consumption_Ratings.csv")

# Data Exploration

df.head()
df.describe().T
df.isnull().any()

df_ = df[['Engine Size','Cylinders','Fuel Consumption', 'CO2 Emissions']]
df_.head()

df_.hist()
plt.show()

# plot each of these features:

plt.scatter(df_['Fuel Consumption'], df_['CO2 Emissions'], color='red')
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Engine Size'], df_['CO2 Emissions'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Cylinders'], df_['CO2 Emissions'], color='black')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()

# Creating train and test data

# first create a model of the linear relationship we think exists between Fuel consumption and CO2 emissions,
# then evaluate the regression with a graph:

X = df_[['Fuel Consumption']]
y = df_[['CO2 Emissions']]

# Modelling

reg_model = LinearRegression().fit(X, y)

# The coefficients

print ('Intercept: ',reg_model.intercept_)    #17.41832581
print ('Coefficients: ', reg_model.coef_[0])  #21.80188686


# Prediction

# How much CO2 Emissions would be expected with 26.10 units of Fuel Consumption?

reg_model.intercept_[0] + reg_model.coef_[0][0]*26.10
#586.4475729069786


# Plot outputs

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9},
                 ci=False, color='r')  #güven aralığı false, yani ekleme
g.set_title(f'Model Equation: CO2 Emissions = {round(reg_model.intercept_[0], 2)} + Fuel*{round(reg_model.coef_[0][0], 2)}')
g.set_ylabel('CO2 Emissions')
g.set_xlabel('Fuel Consumption')
plt.show()

# Evaluation of the model

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 229.99811521450823
y.mean() #260.11
y.std() #64.78

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 15.165688748438306

# MAE
mean_absolute_error(y, y_pred)
# 6.258727993638971

# R-SQUARED
reg_model.score(X, y)
# 0.9451350821741584



# Solution 2: Multiple Regression Model

# This solution of the example of multiple linear regression is predicting;
# CO2 Emission using the features Fuel Consumption, Engine Size and Cylinders of cars.

df = pd.read_csv(r"C:\Users\güneş market\Desktop\exceld\Fuel_Consumption_Ratings.csv")
df.head()
df.describe().T
df_ = df[['Engine Size','Cylinders','Fuel Consumption', 'CO2 Emissions']]
df_.head()
df_.describe().T

X = df_.drop('CO2 Emissions', axis=1)   # drop the co2 emission variable from X
y = df_[["CO2 Emissions"]]
df_.head()


# Creating train and test data
# Around 80% of the entire data will be used for training and 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
y_test.shape
y_train.shape
reg_model = LinearRegression().fit(X_train, y_train)

# The coefficients
print ('Intercept: ',reg_model.intercept_)    #21.96963797
print ('Coefficients: ', reg_model.coef_[0])  #1.12060648,  3.17912023, 19.4271568


# Prediction

# What is the expected value of CO2 Emission based on the following observation values?

# Engine Size: 8.00
# Cylinders: 16.00
# Fuel Consumption: 26.10

# CO2 Emissions  = 21.96963797 + Engine Size * 1.12060648 + Cylinders * 3.17912023 + Fuel Consumption * 19.4271568
21.96963797 + 8.00 * 1.12060648 + 16 * 3.17912023 + 26.10 * 19.4271568 # 588.84920597

new_data_1 = [[8.00], [16.00], [26.10]]
new_data_2 = pd.DataFrame(new_data).T
reg_model.predict(new_data)


# Evaluation of the model

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # 15.446625976286821

# TRAIN RSQUARED
reg_model.score(X_train, y_train) # 0.9423805674075392

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 11.090925681387699

# TEST RSQUARED
reg_model.score(X_test, y_test)  # 0.972015617095251


# We separated the data set as a train test with the hold out method and built a model on the train set,
# and evaluated error in the test set.
# Cross Validation method can also be used!

# 10-Fold Cross Validation and Obtaining RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# np.sqrt = array([4.91609617, 9.77626905, 22.8804846, 29.09053194, 22.98797036, 8.5445722, 6.00530379, 4.79657855, 7.88330539, 8.08132796])
# mean value = 12.496244001683102

# When the data size is small it may be more accurate to use a 10-fold CV or even 5-fold!



# To get better accuracy try to use a multiple linear regression with the same data,
# but this time use 'Fuel Consumption City', 'Fuel Consumption Hwy' instead of 'Fuel Consumption'.

df = pd.read_csv(r"C:\Users\güneş market\Desktop\exceld\Fuel_Consumption_Ratings.csv")
df.head()
df.describe().T
df1_ = df[['Engine Size','Cylinders','Fuel Consumption City', 'Fuel Consumption Hwy', 'CO2 Emissions']]
df1_.head()
df1_.describe().T

X = df1_.drop('CO2 Emissions', axis=1)   # drop the co2 emission variable from X
y = df1_[["CO2 Emissions"]]
df1_.head()


# Creating train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
y_test.shape
y_train.shape
reg_model = LinearRegression().fit(X_train, y_train)

# The coefficients
print ('Intercept: ',reg_model.intercept_)    #21.59204887
print ('Coefficients: ', reg_model.coef_[0])  #1.0463683,  3.23835864, 10.53429237, 8.97245603


# Prediction

# What is the expected value of CO2 Emission based on the following observation values?

# Engine Size: 8.00
# Cylinders: 16.00
# Fuel Consumption City: 30.30
# Fuel Consumption Hwy: 20.90

# CO2 Emissions  = 21.96963797 + Engine Size * 1.12060648 + Cylinders * 3.17912023 + Fuel Consumption * 19.4271568
21.59204887 + 8.00 * 1.0463683 + 16 * 3.23835864 + 30.30 * 10.53429237 + 20.90 * 8.97245603  # 588.490123348

new_data_2 = [[8.00], [16.00], [30.30], [20.90]]
new_data_2 = pd.DataFrame(new_data_2).T
reg_model.predict(new_data_2)


# Evaluation of the model

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # 15.448257728971507

# TRAIN RSQUARED
reg_model.score(X_train, y_train) # 0.9423683931463056

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 11.043899812345115

# TEST RSQUARED
reg_model.score(X_test, y_test)  # 0.9722524232995987

