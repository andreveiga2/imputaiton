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
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, training_Y, test_size = 0.1, random_state = 1, shuffle=False)

    return (X_train, X_test, y_train, y_test)



def create_dummy_set(male = 0, age = 50, time = [2006.583]*10000 , size = 10000):
    """
    Function to create dummy_sets for various functions based on training_variable_type

    Parameters
    ----------
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


    # phi will dictate the length for each array
    # In this case shape of phi is 10000 X 1
    if size == 10000:
        dummy_set['phi'] = np.array(range(10,100001,10))
        dummy_set['yield6'] = np.array([4.66]*size)
        dummy_set['yield12'] = np.array([4.71]*size)
        dummy_set['yield18'] = np.array([4.71]*size)
        dummy_set['yield24'] = np.array([4.71]*size)
        dummy_set['yield30'] = np.array([4.71]*size)
        dummy_set['yield36'] = np.array([4.7]*size)
        dummy_set['yield42'] = np.array([4.69]*size)
        dummy_set['yield48'] = np.array([4.69]*size)
        dummy_set['yield54'] = np.array([4.68]*size)
        dummy_set['yield60'] = np.array([4.67]*size)
        dummy_set['yield66'] = np.array([4.66]*size)
        dummy_set['yield72'] = np.array([4.64]*size)
        dummy_set['yield78'] = np.array([4.63]*size)
        dummy_set['yield84'] = np.array([4.62]*size)
        dummy_set['yield90'] = np.array([4.61]*size)
        dummy_set['yield96'] = np.array([4.6]*size)
        dummy_set['yield102'] = np.array([4.58]*size)
        dummy_set['yield108'] = np.array([4.57]*size)
        dummy_set['yield114'] = np.array([4.56]*size)
        dummy_set['yield120'] = np.array([4.55]*size)
        dummy_set['yield126'] = np.array([4.54]*size)
        dummy_set['yield132'] = np.array([4.53]*size)
        dummy_set['yield138'] = np.array([4.51]*size)
        dummy_set['yield144'] = np.array([4.5]*size)
        dummy_set['yield150'] = np.array([4.49]*size)
        dummy_set['yield156'] = np.array([4.48]*size)
        dummy_set['yield162'] = np.array([4.47]*size)
        dummy_set['yield168'] = np.array([4.46]*size)
        dummy_set['yield174'] = np.array([4.45]*size)
        dummy_set['yield180'] = np.array([4.44]*size)
        dummy_set['yield186'] = np.array([4.42]*size)
        dummy_set['yield192'] = np.array([4.41]*size)
        dummy_set['yield198'] = np.array([4.4]*size)
        dummy_set['yield204'] = np.array([4.38]*size)
        dummy_set['yield210'] = np.array([4.37]*size)
        dummy_set['yield216'] = np.array([4.36]*size)
        dummy_set['yield222'] = np.array([4.34]*size)
        dummy_set['yield228'] = np.array([4.33]*size)
        dummy_set['yield234'] = np.array([4.31]*size)
        dummy_set['yield240'] = np.array([4.3]*size)
        dummy_set['yield246'] = np.array([4.28]*size)
        dummy_set['yield252'] = np.array([4.27]*size)
        dummy_set['yield258'] = np.array([4.25]*size)
        dummy_set['yield264'] = np.array([4.24]*size)
        dummy_set['yield270'] = np.array([4.22]*size)
        dummy_set['yield276'] = np.array([4.21]*size)
        dummy_set['yield282'] = np.array([4.2]*size)
        dummy_set['yield288'] = np.array([4.18]*size)
        dummy_set['yield294'] = np.array([4.17]*size)
        dummy_set['yield300'] = np.array([4.15]*size)

    else:
        dummy_set['phi'] = np.array(csv_data['phi'])
        dummy_set['yield6'] = np.array(csv_data['yield6'])
        dummy_set['yield12'] = np.array(csv_data['yield12'])
        dummy_set['yield18'] = np.array(csv_data['yield18'])
        dummy_set['yield24'] = np.array(csv_data['yield24'])
        dummy_set['yield30'] = np.array(csv_data['yield30'])
        dummy_set['yield36'] = np.array(csv_data['yield36'])
        dummy_set['yield42'] = np.array(csv_data['yield42'])
        dummy_set['yield48'] = np.array(csv_data['yield48'])
        dummy_set['yield54'] = np.array(csv_data['yield54'])
        dummy_set['yield60'] = np.array(csv_data['yield60'])
        dummy_set['yield66'] = np.array(csv_data['yield66'])
        dummy_set['yield72'] = np.array(csv_data['yield72'])
        dummy_set['yield78'] = np.array(csv_data['yield78'])
        dummy_set['yield84'] = np.array(csv_data['yield84'])
        dummy_set['yield90'] = np.array(csv_data['yield90'])
        dummy_set['yield96'] = np.array(csv_data['yield96'])
        dummy_set['yield102'] = np.array(csv_data['yield102'])
        dummy_set['yield108'] = np.array(csv_data['yield108'])
        dummy_set['yield114'] = np.array(csv_data['yield114'])
        dummy_set['yield120'] = np.array(csv_data['yield120'])
        dummy_set['yield126'] = np.array(csv_data['yield126'])
        dummy_set['yield132'] = np.array(csv_data['yield132'])
        dummy_set['yield138'] = np.array(csv_data['yield138'])
        dummy_set['yield144'] = np.array(csv_data['yield144'])
        dummy_set['yield150'] = np.array(csv_data['yield150'])
        dummy_set['yield156'] = np.array(csv_data['yield156'])
        dummy_set['yield162'] = np.array(csv_data['yield162'])
        dummy_set['yield168'] = np.array(csv_data['yield168'])
        dummy_set['yield174'] = np.array(csv_data['yield174'])
        dummy_set['yield180'] = np.array(csv_data['yield180'])
        dummy_set['yield186'] = np.array(csv_data['yield186'])
        dummy_set['yield192'] = np.array(csv_data['yield192'])
        dummy_set['yield198'] = np.array(csv_data['yield198'])
        dummy_set['yield204'] = np.array(csv_data['yield204'])
        dummy_set['yield210'] = np.array(csv_data['yield210'])
        dummy_set['yield216'] = np.array(csv_data['yield216'])
        dummy_set['yield222'] = np.array(csv_data['yield222'])
        dummy_set['yield228'] = np.array(csv_data['yield228'])
        dummy_set['yield234'] = np.array(csv_data['yield234'])
        dummy_set['yield240'] = np.array(csv_data['yield240'])
        dummy_set['yield246'] = np.array(csv_data['yield246'])
        dummy_set['yield252'] = np.array(csv_data['yield252'])
        dummy_set['yield258'] = np.array(csv_data['yield258'])
        dummy_set['yield264'] = np.array(csv_data['yield264'])
        dummy_set['yield270'] = np.array(csv_data['yield270'])
        dummy_set['yield276'] = np.array(csv_data['yield276'])
        dummy_set['yield282'] = np.array(csv_data['yield282'])
        dummy_set['yield288'] = np.array(csv_data['yield288'])
        dummy_set['yield294'] = np.array(csv_data['yield294'])
        dummy_set['yield300'] = np.array(csv_data['yield300'])

        #print(phi.shape)

    # will need to set initializations based on plot_type
    dummy_set['buy_age'] = np.array([age]*size)
    dummy_set['YM'] = np.array(time)
    dummy_set['male'] = np.array([male]*size)

    dummy_set['decision_gar_0'] = np.array([0]*size)
    dummy_set['decision_gar_5'] = np.array([5]*size)
    dummy_set['decision_gar_10'] = np.array([10]*size)

    dummy_set['sqrt_decision_gar_0'] = (dummy_set['decision_gar_0']+1)**(0.3)
    dummy_set['sqrt_decision_gar_5'] = (dummy_set['decision_gar_5']+1)**(0.3)
    dummy_set['sqrt_decision_gar_10'] = (dummy_set['decision_gar_10']+1)**(0.3)
    dummy_set['sqrt_phi'] = dummy_set['phi']**(0.3)
    dummy_set['sqrt_sum_decision_phi_0'] = ((dummy_set['decision_gar_0']+1) + dummy_set['phi'])**0.3
    dummy_set['sqrt_sum_decision_phi_5'] = ((dummy_set['decision_gar_5']+1) + dummy_set['phi'])**0.3
    dummy_set['sqrt_sum_decision_phi_10'] = ((dummy_set['decision_gar_10']+1) + dummy_set['phi'])**0.3


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
    sub_plot_counter_1 = 0
    sub_plot_counter_2 = 0
    fig, ax = plt.subplots(3, 4, sharex=False, sharey=False)
    inchesForMainPlotPart = 7; inchesForLegend = 0.6; percForMain = inchesForMainPlotPart*1.0/(inchesForMainPlotPart+inchesForLegend); percForLegend = 1.-percForMain
    fig.set_size_inches(15,inchesForMainPlotPart+inchesForLegend); #changes width/height of the figure. VERY IMPORTANT
    fig.set_dpi(100); #changes width/height of the figure.
    fig.subplots_adjust(left=0.05, bottom = 0.1*percForMain + percForLegend, right=0.98, top=0.92*percForMain+percForLegend, wspace=0.25, hspace=0.7*percForMain)
    fig.suptitle("Diagonostic Report" , fontsize=18)#, fontweight='bold')
    sub_plot_title_size = 12

    predicted = trained_model.predict(X_test)
    ax[sub_plot_counter_1,sub_plot_counter_2].scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax[sub_plot_counter_1,sub_plot_counter_2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax[sub_plot_counter_1,sub_plot_counter_2].set_ylabel('Predicted')
    ax[sub_plot_counter_1,sub_plot_counter_2].set_xlabel('Measured')
    acc = accuracy(y_test,predicted)
    ax[sub_plot_counter_1,sub_plot_counter_2].set_title("Predicted vs Measured, Error: "+str(acc), fontsize=sub_plot_title_size, y=1.022)

    sub_plot_counter_1 += 1

    for male_i in [0,1]:
        for age_i in [60,65]:

            (X_plot_0, X_plot_5, X_plot_10, dummy_set) = create_dummy_set(male = male_i, age = age_i, time = [2006.583]*10000, size = 10000)

            plot_prediction_0 = trained_model.predict(X_plot_0)
            plot_prediction_5 = trained_model.predict(X_plot_5)
            plot_prediction_10 = trained_model.predict(X_plot_10)

            # counter for number of predictions for which r_0 < r_5 < r1_0
            count = 0
            for i in np.arange(len(plot_prediction_0)):
                if not ((plot_prediction_0[i] > plot_prediction_5[i]) and (plot_prediction_5[i] > plot_prediction_10[i])):
                        count+=1


            ax[sub_plot_counter_1,sub_plot_counter_2].plot(dummy_set['phi'], plot_prediction_0,label = 'decision_0')
            ax[sub_plot_counter_1,sub_plot_counter_2].plot(dummy_set['phi'], plot_prediction_5,label = 'decision_5')
            ax[sub_plot_counter_1,sub_plot_counter_2].plot(dummy_set['phi'], plot_prediction_10,label = 'decision_10')
            ax[sub_plot_counter_1,sub_plot_counter_2].legend(loc='best')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_xlabel('phi')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_ylabel('Predicted Rates')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_title("male:"+str(male_i)+", age:"+str(age_i)+", count:"+str(count), fontsize=sub_plot_title_size, y=1.022)
            fig.savefig('report.png')
            sub_plot_counter_2 += 1
    sub_plot_counter_1 += 1
    sub_plot_counter_2 = 0
    for male_i in [0,1]:
        for age_i in [60,65]:
            plot_prediction_average_0 = []
            plot_prediction_average_5 = []
            plot_prediction_average_10 = []
            time_values = sorted(set(csv_data['YM']))
            for time_i in time_values:
                (X_plot_0, X_plot_5, X_plot_10, dummy_set) = create_dummy_set(male = male_i, age = age_i, time = [time_i]*38230, size = 38230)

                plot_prediction_0 = trained_model.predict(X_plot_0)
                plot_prediction_5 = trained_model.predict(X_plot_5)
                plot_prediction_10 = trained_model.predict(X_plot_10)
                plot_prediction_average_0.append(np.mean(plot_prediction_0))
                plot_prediction_average_5.append(np.mean(plot_prediction_5))
                plot_prediction_average_10.append(np.mean(plot_prediction_10))



            ax[sub_plot_counter_1,sub_plot_counter_2].plot(time_values, plot_prediction_average_0,label = 'decision_0')
            ax[sub_plot_counter_1,sub_plot_counter_2].plot(time_values, plot_prediction_average_5,label = 'decision_5')
            ax[sub_plot_counter_1,sub_plot_counter_2].plot(time_values, plot_prediction_average_10,label = 'decision_10')
            ax[sub_plot_counter_1,sub_plot_counter_2].legend(loc='best')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_xlabel('time')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_ylabel('Predicted Rates')
            ax[sub_plot_counter_1,sub_plot_counter_2].set_title("male:"+str(male_i)+", age:"+str(age_i), fontsize=sub_plot_title_size, y=1.022)
            fig.savefig('report_1.png')
            sub_plot_counter_2 += 1

def grid_search_mlp(X_train, y_train, X_test, y_test, activation = ['relu'], solver = ['adam'], alpha = [0.0001], hidden_size = [10], learning_rate = [0.01]):
    """
    Function to perform grid search to find optimal hyperparameters given training data and test_data
    There are more hyperparameters that can be added to this function but usually these are enough
    MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

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
    activation: array_like(list)
        activations yu wish to loop on, Default ['relu']
        Possible choices ['identity', 'logistic', 'tanh', 'relu']
    solver: array_like(list), Default ['adam']
        solvers yu wish to loop on
        Possible choices ['lbfgs', 'sgd', 'adam']
    alpha: array_like(list), Default [0.0001]
        Range of alpha you wish to try
    hidden_size: array_like(list), Default [10]
        Range of hidden_size you wish to try
    learning_rate: array_like(list), Default [0.01]
        Range of learning_rate you wish to try

    Returns
    -------
    optimal_params: a dictionary with optimal_params

    Raises
    ------
    None
    """
    # These will contain optimal parameters
    optimal_params = {}
    optimal_params['activation'] = ''
    optimal_params['solver'] = ''
    optimal_params['hidden_size'] = 0
    optimal_params['learning_rate'] = 0
    optimal_params['alpha'] = 0
    optimal_error = 100.0

    # Total number of choices to be checked
    total_choices = len(activation)*len(solver)*len(learning_rate)*len(hidden_size)*len(alpha)

    # Counter for number of choices checked till now
    i = 1

    #loop over all choices
    for hidden_size_k in hidden_size:
        for solver_j in solver:
            for activation_i in activation:
                for learning_rate_l in learning_rate:
                    for alpha_m in alpha:

                        # create an object of MLPRegressor type using given Parameters
                        # shuffle = True, Whether to shuffle samples in each iteration, keep True
                        # random_state = 0, random_state is the seed used by the random number generator, to get consistent results
                        # tol = 0.0001, Tolerance for the optimization.
                        nn = MLPRegressor(
                        hidden_layer_sizes=hidden_size_k,  activation=activation_i, solver=solver_j,
                        learning_rate_init=learning_rate_l, shuffle=True, random_state=0, tol = 0.0001, alpha = alpha_m)

                        # fit model on training data
                        n = nn.fit(X_train, y_train)
                        # predict on test data, cross validation
                        predicted = nn.predict(X_test)

                        acc = accuracy(y_test,predicted)
                        #set optimal_params
                        if optimal_error>acc:
                            optimal_error = acc
                            optimal_params['activation'] = activation_i
                            optimal_params['solver'] = solver_j
                            optimal_params['hidden_size'] = hidden_size_k
                            optimal_params['learning_rate'] = learning_rate_l
                            optimal_params['alpha'] = alpha_m
                        i+=1

                        # To keep track of progress, uncomment the following lines
                        print(i, ' / ', total_choices)
                        # print('Lowest Error: (',optimal_params['activation'], optimal_params['solver'], optimal_params['hidden_size'], optimal_params['learning_rate'], optimal_params['alpha'], optimal_error)

    print('Lowest Error: (',optimal_params['activation'], optimal_params['solver'], optimal_params['hidden_size'], optimal_params['learning_rate'], optimal_params['alpha'], optimal_error)
    return optimal_params


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
    #optimal_params = grid_search_mlp(X_train, y_train, X_test, y_test, activation = ['relu'], solver = ['adam'], alpha = np.arange(0.0001, 0.0015, 0.0001), hidden_size = [100,200,300,400,500,600,700,800,900,1000], learning_rate = np.arange(0.01,0.14,0.01))

    optimal_params = {}
    optimal_params['activation'] = 'relu'
    optimal_params['solver'] = 'adam'
    optimal_params['hidden_size'] = 500
    optimal_params['learning_rate'] = 0.04
    optimal_params['alpha'] = 0.0014000000000000002

    #Lowest Error: ( relu adam 500 0.04 0.0014000000000000002 3.5272681821052982
    # call MLP with optimal_params
    nn = MLPRegressor(
    hidden_layer_sizes=optimal_params['hidden_size'], activation=optimal_params['activation'], solver=optimal_params['solver'],
    learning_rate_init=optimal_params['learning_rate'], shuffle=True, random_state=0, alpha = optimal_params['alpha'])

    # fit model on training data
    n = nn.fit(X_train, y_train)

    # (X_plot_0, X_plot_5, X_plot_10, dummy_set) = create_dummy_set(training_variables_type = 3, plot_type = 2)
    #
    # plot_prediction_0 = nn.predict(X_plot_0)
    # plot_prediction_5 = nn.predict(X_plot_5)
    # plot_prediction_10 = nn.predict(X_plot_10)
    #
    # # counter for number of predictions for which r_0 < r_5 < r1_0
    # count = 0
    # for i in np.arange(len(plot_prediction_0)):
    #     if not ((plot_prediction_0[i] > plot_prediction_5[i]) and (plot_prediction_5[i] > plot_prediction_10[i])):
    #             count+=1
    #
    # print(str(count)+" out of "+str(len(plot_prediction_0))+" predictions have NOT (r_0 > r_5 > r_10) ")
    #
    plot(trained_model = nn, X_test = X_test, y_test = y_test)
