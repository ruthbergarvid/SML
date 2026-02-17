import sklearn.discriminant_analysis as skl_da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_ms

n_folds = 10

data = pd.read_csv('training_data_VT2026.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()


output_labels = 'increase_stock'
X = data.drop(columns=[output_labels])
Y = data[output_labels]


def test_model_cross_validation(X, Y, n_folds, threshold, solverr='svd', model):
    cv = skl_ms.KFold(n_splits=n_folds, random_state=1, shuffle=True)
    models = []
    model.append(skl_da.LinearDiscriminantAnalysis())

  
    missclassification = np.zeros((n_folds, len(models))
    for i, (train_index, val_index) in enumerate(cv.split(X)):
        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

        
        model.fit(x_train, y_train)


        prediction_percent = model.predict_proba(x_val)
        prediction = (prediction_percent[:,0] >= threshold)
        err[i] = np.mean(prediction != (y_val == 'high_bike_demand').values)

    average_error = np.mean(err)
    return average_error
