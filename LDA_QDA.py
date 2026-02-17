import sklearn.discriminant_analysis as skl_da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_ms

n_folds = 10

data = pd.read_csv('training_data_VT2026.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()


#input_labels = [col for col in data.columns if col!='increase_stock']
output_labels = 'increase_stock'
X = data.drop(columns=[output_labels])
X = X.drop(columns=['summertime'])
Y = data[output_labels]

#x_train, x_test, y_train, y_test = skl_ms.train_test_split(data[input_labels], data[output_labels], test_size=0.3)





def test_model_cross_validation(X, Y, n_folds, threshold, solverr='svd'):
    cv = skl_ms.KFold(n_splits=n_folds, random_state=1, shuffle=True)
    err = np.zeros(n_folds)
    for i, (train_index, val_index) in enumerate(cv.split(X)):
        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

        if solverr in ['eigen', 'lsqr']:
            model = skl_da.LinearDiscriminantAnalysis(solver=solverr, shrinkage='auto')
        else:
            model = skl_da.LinearDiscriminantAnalysis(solver=solverr)
        model.fit(x_train, y_train)


        prediction_percent = model.predict_proba(x_val)
        prediction = (prediction_percent[:,0] >= threshold)
        err[i] = np.mean(prediction != (y_val == 'high_bike_demand').values)

    average_error = np.mean(err)
    return average_error



def test_threshold():
    thresholds = np.linspace(0, 1, 11)
    lowest_error = 1
    best_threshold = 0
    for i in thresholds:
        i = round(i,2)
        error = test_model_cross_validation(X, Y, n_folds, i)
        print('For threshold:',i,'we get an error of: ', round(error,5))
        if error < lowest_error:
            lowest_error = error
            best_threshold = i
    return lowest_error, best_threshold

def test_included_data(data):
    output_labels = 'increase_stock'
    droped_parameter = ''
    smallest_error = 1
    for excluded_data in data.columns.tolist():
        X = data.drop(columns=[output_labels])
        if excluded_data != output_labels:
            X = X.drop(columns=[excluded_data])
        Y = data[output_labels]
        new_error = test_model_cross_validation(X,Y,n_folds, 0.6)
        print(new_error)
        if new_error < smallest_error:
            smallest_error = new_error
            droped_parameter = excluded_data
    return smallest_error, droped_parameter

def test_solver(data):
    solvers = ['svd', 'lsqr', 'eigen']
    best_error = 1
    best_solver = ''
    for solv in solvers:
        new_error = test_model_cross_validation(X, Y, n_folds, 0.6, solverr=solv)
        print(new_error)
        if new_error < best_error:
            best_error = new_error
            best_solver = solv
    return best_error, best_solver


lowest_error, best_threshold = test_threshold()
print('The lowest error was:', round(lowest_error,3), 'occuring with a threshold of:', round(best_threshold, 3))

smallest_error, droped_parameter = test_included_data(data)
print(smallest_error, droped_parameter)

best_error, best_solver = test_solver(data)
print(best_error, best_solver)
