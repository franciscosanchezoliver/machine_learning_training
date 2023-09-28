"""
Overfitting and underfitting
----------------------------
Interpreting model complexity is a great way to evaluate performance when utilizing supervised learning. Your aim
is to produce a model that can interpret the relationship between features and the target variable, as well
as generalize well when exposed to new observations.

You will generate accuracy scores for the training and test sets using a KNN classifier with different
n_neighbor values.

 - Try different K values (from 1 to 12) and save the train and test set scores got by each k value.
 - Visualize the model complexity curve to see how performance change as the model becomes less complex.

https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=9
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

churn_data = pd.read_csv("churn_data_modified.csv")

# Features
X = churn_data.drop("churn", axis=1).values
# Target
y = churn_data['churn'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Create a list from 1 to 12. We'll iterate over this list to try the effect of the
# K hyperparameter in the model.
neighbors = range(1, 13)

train_scores = {}
test_scores = {}

for each_k in neighbors:
    # Create a KNN Classifier model
    knn = KNeighborsClassifier(n_neighbors=each_k)

    # Train the model
    knn.fit(X_train, y_train)

    # Get the score got with the train set
    train_score_obtained = knn.score(X_train, y_train)
    train_scores[each_k] = train_score_obtained

    # Get the score got with the test set
    test_score_obtained = knn.score(X_test, y_test)
    test_scores[each_k] = test_score_obtained

print("Train Score:")
for k_value, train_score in train_scores.items():
    print(f"{k_value} => {train_score}")

print("Test Score:")

for k_value, test_score in test_scores.items():
    print(f"{k_value} => {test_score}")

print("Compare train-test scores")
for k_value, train_score in train_scores.items():
    train_score = train_scores[k_value]
    test_score = test_scores[k_value]
    print(f" {k_value} | {train_score} - {test_score} | diff {abs(train_score - test_score)} ")

# Visualize the model complexity curve.
plt.title("KNN: Varying Number of Neighbors")

# Plot the train score obtain
plt.plot(
    neighbors,
    train_scores.values(),
    label='Training Accuracy'
)

# Plot the test score obtain
plt.plot(
    neighbors,
    test_scores.values(),
    label='Test Accuracy'
)

# The X axis represent the K (number of neighbors) used to train the model.
plt.xlabel("Number of Neighbors")

# The y-axis represents the Score obtained.
plt.ylabel("Score")

plt.show()
