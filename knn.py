import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Load data
file = r"C:\Users\arvid\OneDrive\Skrivbord\SMASK\training_data_VT2026.csv"
df = pd.read_csv(file)

df["increase_stock"] = df["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1
})

#Feature list
features = [
    "hour_of_day", "day_of_week", "month", "holiday",
    "weekday", "summertime", "temp", "dew",
    "humidity", "precip", "snow", "snowdepth",
    "windspeed", "cloudcover", "visibility"
]

#Feature ablation (CV on full data)
def evaluate(feature_set):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", skl_nb.KNeighborsClassifier(n_neighbors=7))
    ])
    scores = cross_val_score(
        pipe,
        df[feature_set],
        df["increase_stock"],
        cv=5
    )
    return np.mean(scores)

baseline = evaluate(features)
print("All features:", baseline)

def feature_scores(feature_list):
    to_remove = []
    for feature in feature_list:
        reduced = [f for f in feature_list if f != feature]
        score = evaluate(reduced)
        print(f"Without {feature}: {score:.4f}")
        if score > baseline:
            to_remove.append(feature)
    return to_remove

print("Remove these features:", feature_scores(features))

#Selected features
used_features = [
    "hour_of_day",
    "weekday",
    "temp",
    "precip",
    "visibility"
]

#Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df[used_features],
    df["increase_stock"],
    test_size=0.2,
    stratify=df["increase_stock"],
    random_state=1
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

#Hyperparameter tuning (CV on training only)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", skl_nb.KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(1, 51, 2)),
    "knn__weights": ["uniform", "distance"]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_model = grid.best_estimator_

#Threshold tuning (on training set only)
train_probs = best_model.predict_proba(X_train)[:, 1]

thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    preds = (train_probs >= t).astype(int)
    f1 = f1_score(y_train, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best threshold (from training):", best_thresh)
print("Best training F1 with threshold:", best_f1)

#Apply threshold to test set
test_probs = best_model.predict_proba(X_test)[:, 1]
prediction = (test_probs >= best_thresh).astype(int)

#Precision–Recall curve
precision, recall, _ = precision_recall_curve(y_test, test_probs)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

#Final evaluation
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
recall_score = tp / (tp + fn)
precision_score = tp / (tp + fp)
f1_test = 2 * (precision_score * recall_score) / (precision_score + recall_score)

print(f"\nTest Accuracy: {accuracy:.3f}")
print(f"Test Recall: {recall_score:.3f}")
print(f"Test Precision: {precision_score:.3f}")
print(f"Test F1: {f1_test:.3f}")
