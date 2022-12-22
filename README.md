Using scikit-learn to implement Simple Linear Regression and Multiple Linear Regression
Creating a model, training and testing it and use the model.

I have downloaded a fuel consumption dataset, Fuel_Consumption_Ratings.csv, 
which contains model-specific fuel consumption ratings,
and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. 

Dataset source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01



Creating train and test dataset

Train/Test Split involves splitting the dataset into training and testing sets that are mutually exclusive. 
After which, you train with the training set and test with the testing set. 
This will provide a more accurate evaluation on out-of-sample accuracy,
because the testing dataset is not part of the dataset that have been used to train the model. 
Therefore, it gives us a better understanding of how well our model generalizes on new data.

This means that we know the outcome of each data point in the testing dataset, making it great to test with! 
Since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. 
So, in essence, it is truly an out-of-sample testing.

Simple Regression Model

Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) 
to minimize the 'residual sum of squares' between the actual value y in the dataset, 
and the predicted value yhat using linear approximation.

Modelling

Using sklearn package to model data.

Evaluation

We compare the actual values and predicted values to calculate the accuracy of a regression model. 
Evaluation metrics provide a key role in the development of a model, 
as it provides insight to areas that require improvement.

There are different model evaluation metrics, like MSE here to calculate the accuracy of our model based on the test set:

Mean Absolute Error: 
It is the mean of the absolute value of the errors. 
This is the easiest of the metrics to understand since it’s just average error.

Mean Squared Error (MSE): 
Mean Squared Error (MSE) is the mean of the squared error. 
It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. 
This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.

Root Mean Squared Error (RMSE):
R-squared is not an error, but rather a popular metric to measure the performance of your regression model. 
It represents how close the data points are to the fitted regression line. 
The higher the R-squared value, the better the model fits your data. 
The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).


Multiple Regression Model

There are multiple variables that impact the CO2 Emission. 
When more than one independent variable is present, the process is called multiple linear regression. 

Coefficient and Intercept are the parameters of the fitted line. 
Given that it is a multiple linear regression model with 3 parameters, 
and that the parameters are the intercept and coefficients of the hyperplane, 
sklearn can estimate them from our data. 

Scikit-learn uses plain Ordinary Least Squares method to solve this problem.

Ordinary Least Squares (OLS):

OLS is a method for estimating the unknown parameters in a linear regression model. 
OLS chooses the parameters of a linear function of a set of explanatory variables 
by minimizing the sum of the squares of the differences between the target dependent variable and those predicted 
by the linear function. 
In other words, it tries to minimizes the sum of squared errors (SSE),
or mean squared error (MSE) between the target variable (y) and our predicted output over all samples in the dataset.

OLS can find the best parameters using of the following methods:

Solving the model parameters analytically using closed-form equations
Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)