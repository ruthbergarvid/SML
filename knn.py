import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Load data
file = r"C:\Users\arvid\OneDrive\Skrivbord\SMASK\training_data_VT2026.csv"
df = pd.read_csv(file)

df["increase_stock"] = df["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1
})

features = [
    "hour_of_day", "day_of_week", "month", "holiday",
    "weekday", "summertime", "temp", "dew",
    "humidity", "precip", "snow", "snowdepth",
    "windspeed", "cloudcover", "visibility"
]

#Feature ablation (CV) 
def evaluate(feature_set): 
    pipe = Pipeline([ 
        ("scaler", StandardScaler()), 
        ("knn", skl_nb.KNeighborsClassifier(n_neighbors=7)) 
        ]) 
    scores = cross_val_score(pipe, df[feature_set], df["increase_stock"], 
    cv=5) 
    return np.mean(scores)

baseline = evaluate(features)
print("All features:", baseline)

for feature in features:
    reduced = [f for f in features if f != feature]
    score = evaluate(reduced)
    print(f"Without {feature}: {score:.4f}")

used_features = [
    "hour_of_day", "weekday", "temp", "precip", "visibility"
]

X_train, X_test, y_train, y_test = train_test_split(
    df[used_features],
    df["increase_stock"],
    test_size=0.2,
    stratify=df["increase_stock"],
    random_state=1
)

#Hyperparameter tuning
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", skl_nb.KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(1, 51, 2)),
    "knn__weights": ["uniform", "distance"]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

#Final evaluation on test set
best_model = grid.best_estimator_
prediction = best_model.predict(X_test)

cm = confusion_matrix(y_test, prediction, labels=[1, 0])
tp, fn, fp, tn = cm.ravel()

conf_matrix = pd.DataFrame(
    {
        "Predicted: High (1)": [tp, fp],
        "Predicted: Low (0)": [fn, tn]
    },
    index=["Actual: High (1)", "Actual: Low (0)"]
)

print("\nConfusion matrix:")
print(conf_matrix)

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\nTest Accuracy: {accuracy:.3f}")
