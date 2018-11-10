import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

import sklearn.cluster as cluster
import sklearn.mixture as mixture
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import sklearn.decomposition as decomp
from sklearn.model_selection import train_test_split
import scipy.io as sio
import numpy as np
from scipy import stats

#this dictionary will contain all csv data from excel
csv_data = {}

'''
Docstyle to follow
"""
My numpydoc description of a kind
of very exhautive numpydoc format docstring.

Parameters
----------
first : array_like
    the 1st param name `first`
second :
    the 2nd param
third : {'value', 'other'}, optional
    the 3rd param, by default 'value'

Returns
-------
string
    a value in a string

Raises
------
KeyError
    when a key error
OtherError
    when an other error
"""
'''
def accuracy(y, pred_y):
    #mean absolute percentage error
    acc = sum(abs((y-pred_y)/y))/len(y)
    return (acc)*100;

def load_data(filename):
    """
    Function to load all data from csv in csv_data variable

    Parameters
    ----------
    filename : string, gives the csv filename

    Returns
    -------
    None

    Raises
    ------
    IOError when file cannot be opened
    """

    try:
        data = pd.read_csv(filename)
    except IOError:
        print('Cannot open ', filename)

    for i in list(data):
        csv_data[i] = data[i]

# this can be used to load data in csv_data
#load_data("data_JMP_impute.csv")

def create_training_set(small = True):
    """
    Function to split csv data into a training X and Y

    Parameters
    ----------
    small: bool, Default True,
           if small is True then smaller set of decision variables are used

    Returns
    ----------
    training X: predictor data for training
    training_Y: target for training

    Raises
    ----------
    None
    """
    # two categories for preddictors
    training_variables_smallest = ['decision_gar']
    training_variables_small = ['dd_freq_m', 'dd_freq_q', 'dd_freq_h', 'phi', 'decision_gar']
    training_variables_large = 	['dd_freq_m', 'dd_freq_q', 'dd_freq_h', 'phi', 'decision_gar',
                                'dd_pcode_H', 'dd_pcode_M', 'dd_pcode_L', 'fa', 'buy_age', 'YM', 'male', 'dd_sales_D',
                                'dd_scheme_no', 'yield6', 'yield12', 'yield18', 'yield24', 'yield30', 'yield36', 'yield42',
                                'yield96', 'yield102', 'yield108', 'yield114', 'yield120', 'yield126', 'yield132', 'yield138',
                                'yield144', 'yield150', 'yield156', 'yield162', 'yield168', 'yield174', 'yield180', 'yield186',
                                'yield192', 'yield198', 'yield204', 'yield210', 'yield216', 'yield222', 'yield228', 'yield234',
                                'yield240', 'yield246', 'yield252', 'yield258', 'yield264', 'yield270', 'yield276', 'yield282',
                                'yield288', 'yield294', 'yield300']

    training_X = []
    training_Y = csv_data['rate']

    training_X = csv_data['external']
    for i in training_variables_large:
        training_X = np.column_stack((training_X, csv_data[i]))

    #Accuracy improved when data was scaled to (0,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(training_X)

    scaler = Normalizer().fit(training_X)
    normalizedX = scaler.transform(training_X)

    sc = StandardScaler()
    scaled_X = sc.fit_transform(training_X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, training_Y, test_size = 0.08, random_state = 0)

    return (X_train, X_test, y_train, y_test)

def prediction(X_train, X_test, y_train, y_test):
    activation = ['relu', 'identity', 'logistic', 'tanh']
    solver = ['lbfgs','adam']
    hidden_size = [10,100,500,1000]
    learning_rate = np.arange(0.01,0.15,0.01)
    min_activation = ''
    min_solver = ''
    min_hidden_size = 0
    min_learning_rate = 0
    min_error = 100.0
    total_choices = len(activation)*len(solver)*len(learning_rate)*len(hidden_size)
    i = 1
    for hidden_size_k in hidden_size:
        for solver_j in solver:
            for activation_i in activation:
                for learning_rate_l in learning_rate:
                    nn = MLPRegressor(
                    hidden_layer_sizes=hidden_size_k,  activation=activation_i, solver=solver_j,
                    learning_rate_init=learning_rate_l, shuffle=True, random_state=0, tol = 0.0001)


                    n = nn.fit(X_train, y_train)
                    predicted = nn.predict(X_test)
                    #fig, ax = plt.subplots()
                    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
                    #ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
                    #ax.set_xlabel('Measured')
                    #ax.set_ylabel('Predicted')
                    #plt.show()
                    acc = accuracy(y_test,predicted)
                    if min_error>acc:
                        min_error = acc
                        min_activation = activation_i
                        min_solver = solver_j
                        min_hidden_size = hidden_size_k
                        min_learning_rate = learning_rate_l

                    i+=1

                    print('Prediction Error', i, ' / ', total_choices, ' (',activation_i, solver_j, hidden_size_k, learning_rate_l, acc)
                    print('Lowest Error: (',min_activation, min_solver, min_hidden_size, min_learning_rate, min_error)
                    if solver_j == 'lbfgs':
                        break

    print('Lowest Error: (',min_activation, min_solver, min_hidden_size, min_learning_rate, min_error)


load_data("data_JMP_impute.csv")
(X_train, X_test, y_train, y_test) = create_training_set()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
prediction(X_train, X_test, y_train, y_test)
