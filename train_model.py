import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load CSV files
train = pd.read_csv('dataset/sign_mnist_train.csv')
test = pd.read_csv('dataset/sign_mnist_test.csv')

# Separate features and labels
X_train = train.drop('label', axis=1).values
y_train = train['label'].values

X_test = test.drop('label', axis=1).values
y_test = test['label'].values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test accuracy
pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, pred))
