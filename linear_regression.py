import numpy as np
import pandas
from ggplot import *

"""
In this question, you need to:
1) implement the compute_cost() and gradient_descent() procedures
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma


def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features 
    (input data points) and values (output data points).
    """
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """
    
    cost_history = []
    m = len(features)
    for i in range(num_iterations):
        cost = compute_cost(features,values,theta)
        predicted_values= np.dot(features,theta)
        theta = theta - alpha/ m* np.dot((predicted_values-values),features)
        cost_history += [cost]


    return theta, pandas.Series(cost_history) # leave this line for the grader


def get_predictions(dataframe, numerical_features, categorical_features, values_column, alpha, num_iterations):
    # Select Features (try different features!)
    features = dataframe[numerical_features]
    
    # Add UNIT to features using dummy variables
    for categorical_feature in categorical_features:
        dummy_units = pandas.get_dummies(dataframe[categorical_feature], prefix=categorical_feature.lower())
        features = features.join(dummy_units)
    
    # Values
    values = dataframe[values_column]
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions, theta_gradient_descent, (alpha,cost_history)


def plot_cost_history(alpha, cost_history):
   """This function is for viewing the plot of your cost history.
   You can run it by uncommenting this

       plot_cost_history(alpha, cost_history) 

   call in predictions.
   
   If you want to run this locally, you should print the return value
   from this function.
   """
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )


def compute_r_squared(dataframe, values_column, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.
    
    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''
    data = dataframe[values_column]
    num = ((data-predictions)**2).sum()
    denm = ((data - data.mean())**2).sum()
    r_squared = 1 - (num/denm)
    
    return r_squared


def predict(dataframe, numerical_features, values_column, categorical_features=[], 
    alpha=0.1,num_iterations=75, plot_data=False):

    results, thetas, plot_data = get_predictions(dataframe, numerical_features, categorical_features, 
        values_column, alpha, num_iterations)
    print results
    r2_value = compute_r_squared(dataframe, values_column, results)
    if plot_data:
        return results,thetas, r2_value, plot_data
    return results, thetas, r2_value