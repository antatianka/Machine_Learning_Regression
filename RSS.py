
import pandas as pd
from sklearn import  linear_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
train_data = pd.read_csv("wk3_kc_house_train_data.csv", dtype=dict)
test_data = pd.read_csv("wk3_kc_house_test_data.csv", dtype=dict)
valid_data = pd.read_csv("wk3_kc_house_valid_data.csv", dtype=dict)
def get_residual_sum_of_squares(model, data, outcome):

    predictions = model.predict(data)

    residuals = predictions - outcome

    RSS = (residuals*residuals).sum()

    return(RSS)
def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: int(x)**power)
            
    return poly_dataframe

    RSS_deg = 0

    RSS_min = 1e+20

    RMSE_deg = 0

    RMSE_min = 1e+20

    RMSE_d = {}

    RSS_val = {}

    for degree in range(1, 15+1):

        poly_train = polynomial_dataframe(train_data['sqft_living'], degree)
        
        my_features = poly_train.columns.values

        poly_train['price'] = train_data['price']
        target = poly_train['price']
        poly_model = linear_model.LinearRegression()
        poly_model.fit(my_features, target)

        poly_validation = polynomial_dataframe(valid_data['sqft_living'], degree)

        poly_val['price'] = valid_data['price']
        target_val = poly_val['price'] 

        RSS_val['power_' + str(degree)] = get_residual_sum_of_squares(poly_model, poly_val, target_val)
        print RSS_val['power_' + str(degree)] 

        RMSE_d['power_' + str(degree)] = poly_model.evaluate(poly_val)['rmse'] 
        print RMSE_d['power_' + str(degree)]