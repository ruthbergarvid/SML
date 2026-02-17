import sklearn.discriminant_analysis as skl_da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_ms

data = pd.read_csv('training_data_VT2026.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

output_labels = 'increase_stock'
X = data.drop(columns=[output_labels])
Y = data[output_labels]

X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(X,Y, test_size=0.2, random_state=1)



