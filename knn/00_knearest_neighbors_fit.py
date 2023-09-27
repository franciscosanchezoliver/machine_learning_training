"""
k-Nearest Neighbors: Fit
------------------------
In this exercise, you will build a simple classification model using the churn_df dataset.
The features to use will be "account_length" and "customer_service_calls". The target, "churn",
needs to be a single column with the same number of observations as the feature data.

You will convert the features and the target variable into NumPy arrays, create
an instance of a KNN classifier, and then fit it to the data.


To do:
- Extract the features array, for this case "account_length" and "customer_service_calls".
- Extract the target array, the column "churn".
- Create the KNN model using 6 neighbors.
- Fit the classifier.

"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

churn_data = pd.read_csv('churn_data_modified.csv')

X = churn_data[['account_length', 'customer_service_calls']].values
y = churn_data["churn"].values

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X, y)

print(churn_data)

"""
Now you have fit a KNN classifier, you can use it to predict the label of new data points. All available 
data was used for training, however, fortunately, there are new observations available (X_new). 

You will use your classifier to predict the labels of a set of new data points
"""
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

print("New data point")
print(X_new)

predictions = knn.predict(X_new)
print(f"Predictions done for new data point: {predictions}")
