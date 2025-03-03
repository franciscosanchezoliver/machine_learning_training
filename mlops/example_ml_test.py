import pandas as pd
from sklearn.pipeline import Pipeline


def test_pipeline():
    # Mock data for testing
    X_train = pd.DataFrame(
        {"age": [25, 30, 35, 40], "income": [10000, 20000, 30000, 40000]}
    )
    y_train = pd.Series([0, 0, 1, 1])

    pipeline = Pipeline(
        [
            ("preprocessing", DataPreProcessor()),
            ("model", "LogisticRegression()"),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Generate mock data for testing
    X_test = pd.DataFrame(
        {"age": [25, 30, 35, 40], "income": [10000, 20000, 30000, 40000]}
    )
    y_test = pd.Series([0, 0, 1, 1])

    y_pred = pipeline.predict(X_test)

    # Evaluate pipeline on test data
    accuracy = pipeline.score(X_test, y_test)

    assert accuracy > 0.8, "Error, accuracy is too low"
