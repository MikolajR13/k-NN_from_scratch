from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib as mp
from utils import normalize_dataframe
from knn_classifier import KNNClassifier
from metrics import Metrics

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Shape: {X.shape}")
print(f"Classes: {len(np.unique(y))}")
print(f"Features names: {data.feature_names}")
print(f"Classes names: {data.target_names}")
print(X.iloc[114])
print(y.iloc[114])
# normalization to avoid too big difference in features values

X = normalize_dataframe(X)

print(X.iloc[114])
print(y.iloc[114])

# hope this is legal (it's not using scikit-learn for k-NN algorithm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, shuffle=True, stratify=y)

# Shapes of dataframes
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

knn = KNNClassifier(k=3)
y_pred = knn(X_test, X_train, y_train)

print(y_pred)
print(y_test.values)

metrics = Metrics()
results = metrics(y_test, y_pred, 'macro')
print(results)
