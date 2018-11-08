import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

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

    print(csv_data)

# this can be used to load data in csv_data
#load_data("data_JMP_impute.csv")
