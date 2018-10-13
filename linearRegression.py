
### Logistic Regression

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

import os

cwd = os.getcwd()



# THE THREE CONDITIONS:
# 1. The error variable is normally distributed.
# 2. The error variance is constant for all values of x.
# 3. The errors are independent of each other.

########## Part 1 ##########

hwdata = pd.read_csv(os.path.join(cwd, 'height_weight1.csv'))
hwdata.columns = ['height', 'weight']
hwdata.head()

### VARIABLE DESCRIPTIONS

# Height - in inches
# Weight - in lbs

hwdata.describe()

### Split data into training, test

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=5203)
plt.plot(hwdataTrain[['height']], hwdataTrain[['weight']], ".")

### Creating Intercept Model
int_model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
int_model.summary()

### Test residuals
# Test for normalality of residuals
plt.hist(int_model.resid, 50)
# Residuals seem to be normal, with mean 0.

# Test for heteroscedasticity
plt.plot(int_model.predict(hwdataTrain), int_model.resid, '.')
# The prediction plot seems to show constant variance, with no heteroscedasticity.

### Creating No Intercept Model
no_int_model = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
no_int_model.summary()

### Test our residuals
# Test for normalality of residuals
plt.hist(no_int_model.resid, 50)
plt.show()

# Test for heteroscedasticity
plt.plot(no_int_model.predict(hwdataTrain), no_int_model.resid, '.')
# The prediction plot seems to show constant variance, meaning there is no heteroscedasticity.


### Analysis:
# These two models are pretty similar, when looking at our 3 residual
# assumptions. Looking at both residual histogram plots, they both seem to be
# normal, with a mean of 0. The error variance is also constant for all values
# of x, as we can see from the plot of predicted values, meaning that there is
# no heteroscedasticity. And there are no abnormal clumps of data in the
# separate plots, meaning that the data is independent. If we were to use R^2 to
# determine which model fit the data better, we would use the model with no
# intercept, as that model has an R^2 value close to 1 (R^2 = 0.991), meaning
# that the model's fit is better with our data.


########## Part 2 ##########

hwdata2 = pd.read_csv(os.path.join(cwd, 'height_weight2.csv'))
hwdata2.columns = ['height', 'weight']

hwdata2Train, hwdata2Test = train_test_split(hwdata2, test_size=.3, random_state=4210)

no_int_model2 = smf.ols(formula='weight ~ 0 + height', data=hwdata2Train).fit()
no_int_model2.summary()

# Normality of Residuals
plt.hist(no_int_model2.resid, 50)
# Residuals seem to be normal, with mean 0.

# Testing heteroscedasticity
plt.plot(no_int_model2.predict(hwdata2Train), no_int_model.resid, '.')
# Variance seems to be equal over all x values.

### Analysis:
# The model meets the 3 residual assumptions. The residuals seem to be normal,
# as shown by the histogram of the residuals. The variance also seems to be
# pretty equal among all x values, and each value doesn't seem to be affected by
# the others, meaning that the residuals are independent. If one of these were
# not met, we wouldn't be able to confidently use the model, since the
# prediction interval and margin of error would be wider. Point prediction would
# also be inaccurate, since our error margin would be wide and the prediction
# interval would also be bigger.


########## Part 3 ##########

cardata = pd.read_csv(os.path.join(cwd, 'car.csv'))
cardataTrain, cardataTest = train_test_split(cardata, test_size=.3, random_state=123)

car_model = smf.ols(formula='Price ~ 1 + Age +  Miles + C(Make) + C(Type)', data=cardataTrain).fit()
car_model.summary()

car_model2 = smf.ols(formula='np.log(Price) ~ 1 + np.log(Age) +  np.log(Miles) + C(Make) + C(Type)', data=cardataTrain).fit()
car_model2.summary()

car_model3 = smf.ols(formula='Price ~ 1 + np.log(Age) +  np.log(Miles) + C(Make) + C(Type)', data=cardataTrain).fit()
car_model3.summary()


### Analysis:
# On Car Model 2, I used the log of each quantitative variable to create the model. This resulted in a higher R^2, as compared to the other models. The predicted price of the given car is $25,661.29.
