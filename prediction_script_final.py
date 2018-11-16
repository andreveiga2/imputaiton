"""
Code by Chaitanya Baweja
Created: 8th November
Updated: 12th November
"""
#All Import Libraries
import numpy as np # Handling matrix operations
import matplotlib.pyplot as plt #all plotting functions
import pandas as pd #reading and writing from csv
from sklearn.preprocessing import StandardScaler #for scaling data
from sklearn.neural_network import MLPRegressor #mlp regressor used for prediction
from sklearn.model_selection import train_test_split #splitting data into train and test set

#this dictionary will contain all csv data from excel
csv_data = {}

def accuracy(y, pred_y):
    """
    Function to calculate mean absolute percentage error

    Parameters
    ----------
    y : array_like (list)
        contains ground_truth
    pred_y : array_like (list)
             predictions from model
    Returns
    -------
    a float giving mean absolute percentage error

    Raises
    ------
    ZeroDivisionError
        when y has a zero
    """
    try:
        acc = sum(abs((y-pred_y)/y))/len(y)
        return (acc)*100;
    except ZeroDivisionError:
        print("Rate can't be zero", y)


def load_data(filename):
    """
    Function to load all data from csv in csv_data variable

    Parameters
    ----------
    filename : string, gives the csv filename

    Returns
    -------
    data: pandas dataframe for visualization purposes

    Raises
    ------
    IOError when file cannot be opened
    """

    try:
        data = pd.read_csv(filename)
    except IOError:
        print('Cannot open ', filename)

    #to copy all columns read in data to csv_data dictionary
    for i in list(data):
        csv_data[i] = data[i]

    return data


# we define a global StandardScaler object because we want to use the same scaling
# operation during both training and testing time
scaler = StandardScaler()

def create_training_set():
    """
    Function to split csv data into a training X and Y and test X and Y
    Standardize features by removing the mean and scaling to unit variance
    StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


    Parameters
    ----------
    training_variables_type: int, Default 4,
                             if 1 then use training_variables_smallest
                             if 2 then use training_variables_small
                             if 3 then use training_variable: s_new
                             if 4 then use training_variables_large

    split_percentage: int, Default 0.08,
                      define the test set proportion
                      Example if 0.08, then training set is 92% and test set is 8%
    Returns
    ----------
    X_train: predictor data for training
    y_train: target for training
    X_test: predictor data for testin/g
    y_test: target for testing

    Raises
    ----------
    ValueError if wrong type is given as argument
    """
    global scaler

    training_variables_new = 	['decision_gar','buy_age', 'YM', 'male', 'yield6', 'yield12', 'yield18', 'yield24', 'yield30', 'yield36', 'yield42',
                                'yield48', 'yield54', 'yield60', 'yield66', 'yield72', 'yield78', 'yield84', 'yield90',
                                'yield96', 'yield102', 'yield108', 'yield114', 'yield120', 'yield126', 'yield132', 'yield138',
                                'yield144', 'yield150', 'yield156', 'yield162', 'yield168', 'yield174', 'yield180', 'yield186',
                                'yield192', 'yield198', 'yield204', 'yield210', 'yield216', 'yield222', 'yield228', 'yield234',
                                'yield240', 'yield246', 'yield252', 'yield258', 'yield264', 'yield270', 'yield276', 'yield282',
                                'yield288', 'yield294', 'yield300']

    # the training matrix which will have all features column wise, dimensions = (number of records) X (number of features)
    training_X = []
    # the ground truth or target variable containing rates, dimensions = (number of records) X 1
    training_Y = csv_data['rate']
    # assign which features to populate training_X with dependent upon type entered as argumwnt
    training_X = csv_data['phi']
    training_variables = training_variables_new

    for i in training_variables:
        training_X = np.column_stack((training_X, csv_data[i]))

    sqrt_decision_gar = (np.array(csv_data['decision_gar'])+1)**(0.3)
    sqrt_phi = csv_data['phi']**(0.3)
    sqrt_prod_decision_phi = (np.multiply((np.array(csv_data['decision_gar'])+1),csv_data['phi']))**0.3
    sqrt_sum_decision_phi = ((np.array(csv_data['decision_gar'])+1) + csv_data['phi'])**0.3

    complex_features = [sqrt_decision_gar, sqrt_phi, sqrt_sum_decision_phi]
    for i in complex_features:
        training_X = np.column_stack((training_X, i))


    # Standardize features by removing the mean and scaling to unit variance
    # Accuracy improved when this was done.
    scaled_X = scaler.fit_transform(training_X)

    # Split the data into two sets, of 90% training set versus 10% test set,
    # this is done to check the accuracy of the data on unseen data
    # it may make sense to also plot how the operation does on training data to measure training Accuracy
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, training_Y, test_size = 0.1, random_state = 0, shuffle=False)

    return (X_train, X_test, y_train, y_test)

def create_final_test_set(training_variables_type = 4):
    """
    Function to split csv data into a training X and Y and test X and Y
    Standardize features by removing the mean and scaling to unit variance
    StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


    Parameters
    ----------
    training_variables_type: int, Default 4,
                             if 1 then use training_variables_smallest
                             if 2 then use training_variables_small
                             if 3 then use training_variable: s_new
                             if 4 then use training_variables_large

    split_percentage: int, Default 0.08,
                      define the test set proportion
                      Example if 0.08, then training set is 92% and test set is 8%
    Returns
    ----------
    X_train: predictor data for training
    y_train: target for training
    X_test: predictor data for testin/g
    y_test: target for testing

    Raises
    ----------
    ValueError if wrong type is given as argument
    """
    global scaler

    # only use phi and decision_gar as features
    training_variables_smallest = ['decision_gar']
    # use the first few features
    training_variables_small = ['decision_gar','buy_age', 'YM', 'male']
    # experimental set of features which work quite well
    training_variables_new = 	['decision_gar', 'buy_age', 'YM', 'male', 'yield6', 'yield12', 'yield18', 'yield24', 'yield30', 'yield36', 'yield42',
                                'yield48', 'yield54', 'yield60', 'yield66', 'yield72', 'yield78', 'yield84', 'yield90',
                                'yield96', 'yield102', 'yield108', 'yield114', 'yield120', 'yield126', 'yield132', 'yield138',
                                'yield144', 'yield150', 'yield156', 'yield162', 'yield168', 'yield174', 'yield180', 'yield186',
                                'yield192', 'yield198', 'yield204', 'yield210', 'yield216', 'yield222', 'yield228', 'yield234',
                                'yield240', 'yield246', 'yield252', 'yield258', 'yield264', 'yield270', 'yield276', 'yield282',
                                'yield288', 'yield294', 'yield300']

    # found to produce best results even with what feels like quite some noise, additional features thus seem relevant
    training_variables_large = 	['dd_freq_m', 'dd_freq_q', 'dd_freq_h', 'phi', 'decision_gar',
                                'dd_pcode_H', 'dd_pcode_M', 'dd_pcode_L', 'fa', 'buy_age', 'YM', 'male', 'dd_sales_D',
                                'dd_scheme_no', 'yield6', 'yield12', 'yield18', 'yield24', 'yield30', 'yield36', 'yield42',
                                'yield48', 'yield54', 'yield60', 'yield66', 'yield72', 'yield78', 'yield84', 'yield90',
                                'yield96', 'yield102', 'yield108', 'yield114', 'yield120', 'yield126', 'yield132', 'yield138',
                                'yield144', 'yield150', 'yield156', 'yield162', 'yield168', 'yield174', 'yield180', 'yield186',
                                'yield192', 'yield198', 'yield204', 'yield210', 'yield216', 'yield222', 'yield228', 'yield234',
                                'yield240', 'yield246', 'yield252', 'yield258', 'yield264', 'yield270', 'yield276', 'yield282',
                                'yield288', 'yield294', 'yield300']


    # the training matrix which will have all features column wise, dimensions = (number of records) X (number of features)
    training_X = []
    # the ground truth or target variable containing rates, dimensions = (number of records) X 1
    training_Y = csv_data['rate']
    # assign which features to populate training_X with dependent upon type entered as argumwnt
    try:
        if training_variables_type == 1:
            training_X = csv_data['phi']
            training_variables = training_variables_smallest
        else:
            training_X = csv_data['external']
            if training_variables_type == 2:
                training_X = csv_data['phi']
                training_variables = training_variables_small
            elif training_variables_type == 3:
                training_X = csv_data['phi']
                training_variables = training_variables_new
            elif training_variables_type == 4:
                training_variables = training_variables_large
            else:
                raise ValueError('Type can only be between 1 - 4, type specified: ', training_variables_type)
    except ValueError as e:
        print(e.args)

    decision_0 = np.array([0]*38230)
    decision_5 = np.array([5]*38230)
    decision_10 = np.array([10]*38230)

    for i in training_variables:
        training_X = np.column_stack((training_X, csv_data[i]))

    sqrt_decision_gar = (np.array(csv_data['decision_gar'])+1)**(0.3)
    sqrt_phi = csv_data['phi']**(0.3)
    sqrt_prod_decision_phi = (np.multiply((np.array(csv_data['decision_gar'])+1),csv_data['phi']))**0.3
    sqrt_sum_decision_phi = ((np.array(csv_data['decision_gar'])+1) + csv_data['phi'])**0.3


    #complex_features = [sqrt_decision_gar, sqrt_phi, sqrt_prod_decision_phi, sqrt_sum_decision_phi]
    complex_features = [sqrt_decision_gar, sqrt_phi, sqrt_sum_decision_phi]
    for i in complex_features:
        training_X = np.column_stack((training_X, i))

    final_test_0 = training_X

    # Standardize features by removing the mean and scaling to unit variance
    # Accuracy improved when this was done.
    scaled_X = scaler.fit_transform(training_X)

    # Split the data into two sets, of 92% training set versus 8% test set,
    # this is done to check the accuracy of the data on unseen data
    # it may make sense to also plot how the operation does on training data to measure training Accuracy
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, training_Y, test_size = split_percentage, random_state = 0, shuffle=False)

    return (X_train, X_test, y_train, y_test)


def create_dummy_set(training_variables_type = 3, plot_type = 2):
    """
    Function to create dummy_sets for various functions based on training_variable_type

    Parameters
    ----------
    trained_model : model to be used for predictions
    X_test : array_like(list)
        Matrix with test data predictors
    y_train : array_like(list)
        Matrix(Column) with test data target
    training_variables_type: int, Default 4,
                             if 1 then use training_variables_smallest
                             if 2 then use training_variables_small
                             if 3 then use training_variable: s_new
                             if 4 then use training_variables_large
    plot_type: int, Default 1,
        if 1 then no dummy data required

    Returns
    -------
    X_plot_0, X_plot_5, X_plot_10:
        Dummy predictor data corresponding to each decision_gar

    Raises
    ------
    ValueError
        when a model_type that has not been implemented is input
    """
    dummy_set = {}

    if plot_type == 2:
        # phi will dictate the length for each array
        # In this case shape of phi is 38230 X 1
        dummy_set['phi'] = np.array(sorted(csv_data['phi']))
        #print(phi.shape)

        # will need to set initializations based on plot_type
        dummy_set['buy_age'] = np.array([50]*38230)
        dummy_set['YM'] = np.array([2006.583]*38230)
        dummy_set['male'] = np.array([0]*38230)

        dummy_set['decision_gar_0'] = np.array([0]*38230)
        dummy_set['decision_gar_5'] = np.array([5]*38230)
        dummy_set['decision_gar_10'] = np.array([10]*38230)

        dummy_set['sqrt_decision_gar_0'] = (dummy_set['decision_gar_0']+1)**(0.3)
        dummy_set['sqrt_decision_gar_5'] = (dummy_set['decision_gar_5']+1)**(0.3)
        dummy_set['sqrt_decision_gar_10'] = (dummy_set['decision_gar_10']+1)**(0.3)
        dummy_set['sqrt_phi'] = dummy_set['phi']**(0.3)
        dummy_set['sqrt_sum_decision_phi_0'] = ((dummy_set['decision_gar_0']+1) + dummy_set['phi'])**0.3
        dummy_set['sqrt_sum_decision_phi_5'] = ((dummy_set['decision_gar_5']+1) + dummy_set['phi'])**0.3
        dummy_set['sqrt_sum_decision_phi_10'] = ((dummy_set['decision_gar_10']+1) + dummy_set['phi'])**0.3

        dummy_set['yield6'] = np.array([4.66]*38230)
        dummy_set['yield12'] = np.array([4.71]*38230)
        dummy_set['yield18'] = np.array([4.71]*38230)
        dummy_set['yield24'] = np.array([4.71]*38230)
        dummy_set['yield30'] = np.array([4.71]*38230)
        dummy_set['yield36'] = np.array([4.7]*38230)
        dummy_set['yield42'] = np.array([4.69]*38230)
        dummy_set['yield48'] = np.array([4.69]*38230)
        dummy_set['yield54'] = np.array([4.68]*38230)
        dummy_set['yield60'] = np.array([4.67]*38230)
        dummy_set['yield66'] = np.array([4.66]*38230)
        dummy_set['yield72'] = np.array([4.64]*38230)
        dummy_set['yield78'] = np.array([4.63]*38230)
        dummy_set['yield84'] = np.array([4.62]*38230)
        dummy_set['yield90'] = np.array([4.61]*38230)
        dummy_set['yield96'] = np.array([4.6]*38230)
        dummy_set['yield102'] = np.array([4.58]*38230)
        dummy_set['yield108'] = np.array([4.57]*38230)
        dummy_set['yield114'] = np.array([4.56]*38230)
        dummy_set['yield120'] = np.array([4.55]*38230)
        dummy_set['yield126'] = np.array([4.54]*38230)
        dummy_set['yield132'] = np.array([4.53]*38230)
        dummy_set['yield138'] = np.array([4.51]*38230)
        dummy_set['yield144'] = np.array([4.5]*38230)
        dummy_set['yield150'] = np.array([4.49]*38230)
        dummy_set['yield156'] = np.array([4.48]*38230)
        dummy_set['yield162'] = np.array([4.47]*38230)
        dummy_set['yield168'] = np.array([4.46]*38230)
        dummy_set['yield174'] = np.array([4.45]*38230)
        dummy_set['yield180'] = np.array([4.44]*38230)
        dummy_set['yield186'] = np.array([4.42]*38230)
        dummy_set['yield192'] = np.array([4.41]*38230)
        dummy_set['yield198'] = np.array([4.4]*38230)
        dummy_set['yield204'] = np.array([4.38]*38230)
        dummy_set['yield210'] = np.array([4.37]*38230)
        dummy_set['yield216'] = np.array([4.36]*38230)
        dummy_set['yield222'] = np.array([4.34]*38230)
        dummy_set['yield228'] = np.array([4.33]*38230)
        dummy_set['yield234'] = np.array([4.31]*38230)
        dummy_set['yield240'] = np.array([4.3]*38230)
        dummy_set['yield246'] = np.array([4.28]*38230)
        dummy_set['yield252'] = np.array([4.27]*38230)
        dummy_set['yield258'] = np.array([4.25]*38230)
        dummy_set['yield264'] = np.array([4.24]*38230)
        dummy_set['yield270'] = np.array([4.22]*38230)
        dummy_set['yield276'] = np.array([4.21]*38230)
        dummy_set['yield282'] = np.array([4.2]*38230)
        dummy_set['yield288'] = np.array([4.18]*38230)
        dummy_set['yield294'] = np.array([4.17]*38230)
        dummy_set['yield300'] = np.array([4.15]*38230)


        X_plot_0 = dummy_set['phi']
        X_plot_5 = dummy_set['phi']
        X_plot_10 = dummy_set['phi']
        plot_variables_0 = [dummy_set['decision_gar_0'], dummy_set['buy_age'], dummy_set['YM'], dummy_set['male'], dummy_set['yield6'], dummy_set['yield12'], dummy_set['yield18'], dummy_set['yield24'], dummy_set['yield30'], dummy_set['yield36'], dummy_set['yield42'],
                            dummy_set['yield48'], dummy_set['yield54'], dummy_set['yield60'], dummy_set['yield66'], dummy_set['yield72'], dummy_set['yield78'], dummy_set['yield84'], dummy_set['yield90'],
                            dummy_set['yield96'], dummy_set['yield102'], dummy_set['yield108'], dummy_set['yield114'], dummy_set['yield120'], dummy_set['yield126'], dummy_set['yield132'], dummy_set['yield138'],
                            dummy_set['yield144'], dummy_set['yield150'], dummy_set['yield156'], dummy_set['yield162'], dummy_set['yield168'], dummy_set['yield174'], dummy_set['yield180'], dummy_set['yield186'],
                            dummy_set['yield192'], dummy_set['yield198'], dummy_set['yield204'], dummy_set['yield210'], dummy_set['yield216'], dummy_set['yield222'], dummy_set['yield228'], dummy_set['yield234'],
                            dummy_set['yield240'], dummy_set['yield246'], dummy_set['yield252'], dummy_set['yield258'], dummy_set['yield264'], dummy_set['yield270'], dummy_set['yield276'], dummy_set['yield282'],
                            dummy_set['yield288'], dummy_set['yield294'], dummy_set['yield300'], dummy_set['sqrt_decision_gar_0'], dummy_set['sqrt_phi'], dummy_set['sqrt_sum_decision_phi_0'] ]


        plot_variables_5 = [dummy_set['decision_gar_5'], dummy_set['buy_age'], dummy_set['YM'], dummy_set['male'], dummy_set['yield6'], dummy_set['yield12'], dummy_set['yield18'], dummy_set['yield24'], dummy_set['yield30'], dummy_set['yield36'], dummy_set['yield42'],
                            dummy_set['yield48'], dummy_set['yield54'], dummy_set['yield60'], dummy_set['yield66'], dummy_set['yield72'], dummy_set['yield78'], dummy_set['yield84'], dummy_set['yield90'],
                            dummy_set['yield96'], dummy_set['yield102'], dummy_set['yield108'], dummy_set['yield114'], dummy_set['yield120'], dummy_set['yield126'], dummy_set['yield132'], dummy_set['yield138'],
                            dummy_set['yield144'], dummy_set['yield150'], dummy_set['yield156'], dummy_set['yield162'], dummy_set['yield168'], dummy_set['yield174'], dummy_set['yield180'], dummy_set['yield186'],
                            dummy_set['yield192'], dummy_set['yield198'], dummy_set['yield204'], dummy_set['yield210'], dummy_set['yield216'], dummy_set['yield222'], dummy_set['yield228'], dummy_set['yield234'],
                            dummy_set['yield240'], dummy_set['yield246'], dummy_set['yield252'], dummy_set['yield258'], dummy_set['yield264'], dummy_set['yield270'], dummy_set['yield276'], dummy_set['yield282'],
                            dummy_set['yield288'], dummy_set['yield294'], dummy_set['yield300'], dummy_set['sqrt_decision_gar_5'], dummy_set['sqrt_phi'], dummy_set['sqrt_sum_decision_phi_5'] ]

        plot_variables_10 = [dummy_set['decision_gar_10'], dummy_set['buy_age'], dummy_set['YM'], dummy_set['male'], dummy_set['yield6'], dummy_set['yield12'], dummy_set['yield18'], dummy_set['yield24'], dummy_set['yield30'], dummy_set['yield36'], dummy_set['yield42'],
                            dummy_set['yield48'], dummy_set['yield54'], dummy_set['yield60'], dummy_set['yield66'], dummy_set['yield72'], dummy_set['yield78'], dummy_set['yield84'], dummy_set['yield90'],
                            dummy_set['yield96'], dummy_set['yield102'], dummy_set['yield108'], dummy_set['yield114'], dummy_set['yield120'], dummy_set['yield126'], dummy_set['yield132'], dummy_set['yield138'],
                            dummy_set['yield144'], dummy_set['yield150'], dummy_set['yield156'], dummy_set['yield162'], dummy_set['yield168'], dummy_set['yield174'], dummy_set['yield180'], dummy_set['yield186'],
                            dummy_set['yield192'], dummy_set['yield198'], dummy_set['yield204'], dummy_set['yield210'], dummy_set['yield216'], dummy_set['yield222'], dummy_set['yield228'], dummy_set['yield234'],
                            dummy_set['yield240'], dummy_set['yield246'], dummy_set['yield252'], dummy_set['yield258'], dummy_set['yield264'], dummy_set['yield270'], dummy_set['yield276'], dummy_set['yield282'],
                            dummy_set['yield288'], dummy_set['yield294'], dummy_set['yield300'], dummy_set['sqrt_decision_gar_10'], dummy_set['sqrt_phi'], dummy_set['sqrt_sum_decision_phi_10'] ]

        # build X_plots based on variables specified in plot_vatiables
        for i in plot_variables_0:
            X_plot_0 = np.column_stack((X_plot_0, i))
        for i in plot_variables_5:
            X_plot_5 = np.column_stack((X_plot_5, i))
        for i in plot_variables_10:
            X_plot_10 = np.column_stack((X_plot_10, i))

        # perform the same scaler transformation here as was performed on training data
        X_plot_0 = scaler.transform(X_plot_0)
        X_plot_5 = scaler.transform(X_plot_5)
        X_plot_10 = scaler.transform(X_plot_10)

        return (X_plot_0, X_plot_5, X_plot_10, dummy_set)


def plot(trained_model = None, X_test = [], y_test = [], phi = [], plot_prediction_0 = [], plot_prediction_5 = [],
    plot_prediction_10 = [], plot_type = 1):
    """
    Function to plot a set of graphs using trained_model and test data
    For some graphs, we will need dummy data for which a create_dummy_set() will be called from this code

    Parameters
    ----------
    trained_model : model to be used for predictions
    X_test : array_like(list)
        Matrix with test data predictors
    y_test : array_like(list)
        Matrix(Column) with test data target
    plot_type: int, Default 1,
        if 1 then plot prediction vs actual values

    Returns
    -------
    plots

    Raises
    ------
    ValueError
        when a model_type that has not been implemented is input
    """
    if plot_type == 1:
        predicted = trained_model.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        acc = accuracy(y_test,predicted)
    elif plot_type == 2:
        fig, ax = plt.subplots()
        ax.plot(phi, plot_prediction_0,label = 'decision_0')
        ax.plot(phi, plot_prediction_5,label = 'decision_5')
        ax.plot(phi, plot_prediction_10,label = 'decision_10')
        ax.legend(loc='best')
        #ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('phi')
        ax.set_ylabel('Predicted')
        plt.show()

def prediction(X_train, y_train, X_test, y_test):
    """
    Function to fit a model for given training set using model type

    Parameters
    ----------
    X_train : array_like(list)
        Matrix with training data predictors
    y_train : array_like(list)
        Matrix(Column) with training data target
    X_test : array_like(list)
        Matrix with test data predictors
    y_test : array_like(list)
        Matrix(Column) with test data target
    model_type : string, Default MLPRegressor
        which model type to use,
        Only MLP implemented in this Code

    Returns
    -------
    model: an object of model_type which has been fit on training data

    Raises
    ------
    ValueError
        when a model_type that has not been implemented is given as argument
    """

    # This is done to use the same scaler as the global one so that scaling operations
    global scaler
    optimal_params = {}
    optimal_params['activation'] = 'relu'
    optimal_params['solver'] = 'adam'
    optimal_params['hidden_size'] = 5
    optimal_params['learning_rate'] = 0.09
    optimal_params['alpha'] = 0.0008

    # call MLP with optimal_params
    nn = MLPRegressor(
    hidden_layer_sizes=optimal_params['hidden_size'], activation=optimal_params['activation'], solver=optimal_params['solver'],
    learning_rate_init=optimal_params['learning_rate'], shuffle=True, random_state=0, tol = 0.0001, alpha = optimal_params['alpha'])

    # fit model on training data
    n = nn.fit(X_train, y_train)

    (X_plot_0, X_plot_5, X_plot_10, dummy_set) = create_dummy_set(training_variables_type = 3, plot_type = 2)

    plot_prediction_0 = nn.predict(X_plot_0)
    plot_prediction_5 = nn.predict(X_plot_5)
    plot_prediction_10 = nn.predict(X_plot_10)

    # counter for number of predictions for which r_0 < r_5 < r1_0
    count = 0
    for i in np.arange(len(plot_prediction_0)):
        if not ((plot_prediction_0[i] > plot_prediction_5[i]) and (plot_prediction_5[i] > plot_prediction_10[i])):
                count+=1

    print(str(count)+" out of "+str(len(plot_prediction_0))+" predictions have NOT (r_0 > r_5 > r_10) ")

    #plotting Measured vs predictor
    plot(trained_model = nn, X_test = X_test, y_test = y_test, plot_type = 1)

    # Predicting rates vs phi with everything constant
    plot(phi = dummy_set['phi'], plot_prediction_0 = plot_prediction_0, plot_prediction_5 = plot_prediction_5,
        plot_prediction_10 = plot_prediction_10, plot_type = 2)
