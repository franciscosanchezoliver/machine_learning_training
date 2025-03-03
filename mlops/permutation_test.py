"""
Feature importance tests
Feature importance tests test the importance of features in an ML model and help 
identify which features are the most important in making predictions. One 
example is permutation importance. The idea is to randomly permute the values 
of features to see how much the performance of the model changes. These kinds 
of tests constantly test a model's sensitivity to features and can inform 
whether it is worth re-training with an updated dataset.

link: https://campus.datacamp.com/courses/developing-machine-learning-models-for-production/testing-ml-pipelines?ex=4
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load iris dataset
iris = load_iris()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

# Train a Random Forest Classifier on the training set
model = RandomForestClassifier().fit(X_train, y_train)

# Then use the "permutation_important" function form sklearn

# Calculate feature importances using permutation importance
results = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=0
)

# Print the feature importances
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
importances = results.importances_mean

for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {importances[i]:.3f}")
