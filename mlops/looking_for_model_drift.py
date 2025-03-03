"""
Models can exhibit signs of drift, or changes, in how it makes predictions 
over time.

Concept Drift
-------------
Concept drift refers to a shift in the relationship between features and the 
response.
This can occur when the meaning or usage of features changes over time.

For example: the meaning of "sick" can be a good thing


Prediction Drift
----------------
Prediction drift refers to a shift in the model's prediction distribution, 
while label drift refers to a shift in the actual label distribution.


Both types of drift can occur when underlying data changes, but measured 
at the model level.

Example: Looking for model drift
--------------------------------
In the following exercise, we are simulating concept drift by introducing 
data drift to X.

link: https://campus.datacamp.com/courses/developing-machine-learning-models-for-production/testing-ml-pipelines?ex=7

"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Create a classifier
model = DecisionTreeClassifier(random_state=0)

# Train a classifier on training data
model.fit(X_train, y_train)


# Predict with the test set labels
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# The addition of noise to the test data is a form of data drift,
# which is a change in the distribution of the input data.
# By introducing this change, we are testing the model's ability to adapt
# to new or unexpected patterns in the input data, which is one of the
# key challenges in dealing with concept drift.

# Simulate data drift
X_test_drift = X_test + 1.0

# Calculate the accuracy on drifted data
y_pred_drift = model.predict(X_test_drift)
accuracy_drift = accuracy_score(y_test, y_pred_drift)
print("Accuracy on drifted data:", accuracy_drift)

# Set a certain threshold to detect if our model suffered from data drift
drift_threshold = accuracy * 0.9

if accuracy_drift < drift_threshold:
    print("Concept drift detected")
else:
    print("No concept drift detected.")
