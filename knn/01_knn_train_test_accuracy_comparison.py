"""
- Use all variables as features
- Divide the data into a train and a test set. The size of the test
  set should be the 20% of the data.
- Target is the binary variable churn/not churn
- Create a KNN model with 5 neighbors
- Print the score got with the test set.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os


churn_data = pd.read_csv(os.getcwd() + "\\knn\\churn_data_modified.csv")

# The features will be all the variables.
X = churn_data.drop('churn', axis=1).values

# The target will tell us if the client has churned or not.
y = churn_data['churn'].values

# Split the data into train and test sets.
# We also want to keep the same proportion of the target labels
# in both sets (stratify).
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Create a model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Print the accuracy with the test set
test_score = knn.score(X_test, y_test)
print(f"Test score: {test_score}")
