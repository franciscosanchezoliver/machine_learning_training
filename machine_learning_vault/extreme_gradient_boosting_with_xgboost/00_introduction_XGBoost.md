#xgboost #decision_tree #boosting 

## Introduction XGB
https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/classification-with-xgboost?ex=1
XGBoost its an implementation of Gradient Boosting, 
In order to understand XGBoost, we need to have some knowledge on the broader
topics of supervised classification, decision tress and boosting.

To begin, lets briefly review what supervised learning is and the kind of 
problems its methods can be applied to.

Supervised learning is the kind of learning problem that XGBoost is applied to, 
which relies on labeled data. That is, you have some understanding of the
past behavior of the problem you're trying to solve or what you're trying 
to predict.

For example, assessing whether a specific image contains a person's face is
a classification problem.
Here the training data are images converted into vectors of pixel values, and
the labels are either 1 when the image contains a face or 0 when the image 
doesn't contain a face.
Given this, there are two kinds of supervised learning problems that account
for the vast majority of use cases: classification and regression.


## Supervised Learning: Classification Problems
Classification problems involve predicting either binary or multi-class
outcomes.
For example, predicting whether a person will purchase an insurance package
given some quote is a binary supervised learning problem.
And predicting whether a picture contains one of several species of birds
is a multi-class supervised learning problem.
When dealing with binary supervised learning problems, the AUC (Area Under
the Receive Operating Characteristic (ROC)) is the most versatile and common
evaluation metric used to judge the quality of a binary classification model.

[Logictic Regression and the ROC curve](../supervised_learning_with_scikit-learn/06_logistic_regression_and_the_roc_curve/06_logistic_regression_and_the_roc_curve.md)

Its simply the probability that a randomly chosen positive data point will have
a higher rank than a randomly chosen negative data point for your learning problem.

When dealing with multiclass classification problems, its common to use the
accuracy score (higher is better) and to look at the overall confusion matrix
to evaluate the quality of a model

$$ accuracy = \frac{t_p + t_n}{t_p + t_n + f_p + f_n}$$

[Measuring how good is your model](../supervised_learning_with_scikit-learn/05_how_good_is_your_model/05_how_good_is_your_classification_model.md)

Some common algorithms for classification problems include logistic regression
and decision trees.

All supervised learning problems, including classification problems, require 
that the data is structured as a table of feature vectors, where the features
themselves (also called attributes or predictors) are either numerical or
categorical.

Furthermore, it is usually the case that the numeric features are scaled to 
aid in either feature interpretation or to ensure that the model can be trained
properly.

For example, numeric feature scaling is essential to ensure property trained
support vector machine models.

Categorical features are also almost always encoded before applying supervised
learning algorithms, most commonly using one-hot encoding.

Finally, other kind of supervised learning problems exist, so I'll mention
them briefly.

Ranking problems involve predicting an ordering on a set of choices (like
google search suggestions) and recommendation problems include recommending 
and item or a set or items to a user based on his/her consumption history 
and profile (like Netflix).

https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/classification-with-xgboost?ex=2
## Exercise 
Which of these is a classification problem?<br>
Given below are 4 potential machine learning problems you might encounter in the wild. Pick the one that is a classification problem.

- [ ] Given past performance of stocks and various other financial data, predicting the exact price of a given stock (Google) tomorrow.
    - Not quite. This is an example of a regression problem, because we are predicting a continuous quantity.

- [ ] Given a large dataset of user behaviors on a website, generating an informative segmentation of the users based on their behaviors.
    - There's nothing to predict here, this is an unsupervised (clustering) problem.

-  [X] Predicting whether a given user will click on an ad given the ad content and metadata associated with the user.

- [ ] Given a user's past behavior on a video platform, presenting him/her with a series of recommended videos to watch next.
    - Incorrect. This problem involves ranking entities and returning the highest ranked ones (in order) to the user.


https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/classification-with-xgboost?ex=3
## Exercise
Which of these is a binary classification problem?<br>
Great! A classification problem involves predicting the category a given data point belongs to out of a finite set of possible categories. Depending on how many possible categories there are to predict, a classification problem can be either binary or multi-class. Let's do another quick refresher here. Your job is to pick the binary classification problem out of the following list of supervised learning problems.


-  [X] Predicting whether a given image contains a cat.
- [ ] Predicting the emotional valence of a sentence (Valence can be positive, negative, or neutral).
    - There are 3 categories to choose from here. Try again.
- [ ] Recommending the most tax-efficient strategy for tax filing in an automated accounting system.
    - This smells like a recommendation problem, not a classification problem.
- [ ] Given a list of symptoms, generating a rank-ordered list of most likely diseases.
    - Incorrect. This is a recommendation problem.

## Introduction
XGBoots its a popular machine learning library for good reason. It was developed
originally as a C++ command-line application. After winning a popular machine
learning competition, the package started being adopted wither the ML 
community. As a result, bindings, or functions that tapped into the core C++
code, started appearing in a variety of other languages, including Python, R, 
Scala, and Julia.
We will cover the Python API in this course.

### What makes XGBoost so popular?
Its speed and performance, because the core XGBoost algorithm is parallelizable
it can harness all the processing power of modern multi-core computers.
Furthermore, it is parallelizable onto GPU's and across networks of computers, 
making it feasible to train models on very large datasets on the order of 
hundreds of millions training examples.

However, XGBoost's speed isn't the package's real draw. Ultimately, a fast but
poorly performing machine learning algorithm is not going to have a wide 
adoption within the community. 
What makes XGBoost so popular is that it consistently outperforms almost all 
others single-algorithms methods in machine learning competitions and has been
shown to achieve state-of-the-art performance on a variety of benchmark 
machine learning datasets.

Here's an example of how we can use XGBoost using classification problem.

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class_data = pd.read_csv("classification_data.csv")

# Separate between features and target
X, y = class_data.iloc[:, :-1], class_data.iloc[:, -1]

# Separate data between train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Instantiate our XGBoost Classifier
xg_cl = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators =10, 
    seed=123
)

# Train the model
xg_cl.fix(X_train, y_train)

# Predict on the test set
preds = xg_cl.predict(X_test)

# Calculate how good our model is
accuracy = float(np.sum(preds == y_test))/y_test.shape[0]

print("accuracy :%f" % (accuracy))
# accuracy: 0.78
```

https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/classification-with-xgboost?ex=6
### What is a decision tree?
Because XGBoost is usually used with trees as base learners, we need to 
understand what an individual decision tree is, and how it works.

Here is one example of a decision tree
```sequence

Road Tested?
  |- No => Don't buy
  |- Si => 
    |
    |- Mileage?
        |- Low => Buy 
        |- High => 
            | -Age?
                | - old => Don't Buy
                | - recent => Buy
```

As you can see, it has a single question that is being asked at each decision
node, and only 2 possible choices, at the very bottom of each decision tree, 
there is a single possible decision.

In this example decision tree for whether to purchase a vehicle, the first 
question you ask is whether it has been road-tested, if it hasn't you immediately
decide not to buy, otherwise, you continue asking questions, such as what the
vehicle's mileage is, and, it its age is old or recent.

At bottom, every possible decision will eventually lead to a choice, some 
taking many fewer questions to get those choices than others.

The concept of a base learner will be covered more extensively later, but for 
now, just think of any individual learning algorithm in a ensemble algorithm 
as a base learner. This is important because XGBoost itself is an ensemble
learning method and it uses the outputs of many models for a final decision.

Anyways, as you saw in the previous slide, decision tree is a learning method 
that involves a tree-like graph to model either a continuous or categorical 
choice given some data.

It's composed of a series of binary decisions (yes/no or true/false) that when
answered in succession ultimately yield a prediction about the data at hand
(these predictions happen at the leaves of the tree).

Decision trees are constructed iteratively (that is, one binary decision 
at a time) util some stopping criterion is met (the depth of the tree
reaches some pre-defined maximum value, for example).

During construction, the tree is built one split at a time, and the way that
a split is selected (that is, what feature to split on and where in the 
feature's range of values to split) can vary, but involves choosing a split 
point that segregates the target values better (that is, put each target category 
into buckets that are increasingly dominated by just one category) until all
or (nearly all) values within a given split are exclusively of one category
or another.

Using this process, each leaf of the decision tree will have category in the majority, or should exclusively of one category.

Individual decision trees in general are low-bias, high-variance learning models.

![high_variance_low_bias](Pasted%20image%2020240601143451.png)

That is, they are very good at learning relationships within any data you train them on, but they tend to overfit the data you use to train them on and usually generalize to new data poorly.

XGBoost uses a slightly different kind of a decision tree, called a classification and regression tree called "CART".

Whereas for the decision trees described above, the leaf nodes always contain decision values, CART trees contain a real-valued scored in each leaf, regardless of whether they are used for classification for regression.

The real values scores can then be thresholded to convert into categories for classification problems if necessary.


### Exercise
[Decision Tree Model example](../../extreme_gradient_boosting_with_xgboost/02_exercise_decision_trees/decision_trees.py)


https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/classification-with-xgboost?ex=8
## What is Boosting?
Now that we've reviewed both supervised learning and the basics of decision trees, 
lets talk about the core concept that gives XGBoost its state-of-the-art 
performance, boosting.

At bottom, boosting isn't really a specific machine learning algorithm, but a 
concept that can be applied to a set of machine learning models.

So, it's really a meta-algorithm. Specifically, it is an ensamble meta-algorithm 
primarily used to reduce any given single learner's variance and to convert 
many weak learners into an arbitrarily strong learner.

A weak learner is any machine learning algorithm that is just slightly better 
than chance.

So, a decision tree that can predict some outcome slightly more frequently than
pure randomness would be considered a weak learner.

The principal insight that allows XGBoost to work is the fact that you can use
boosting to convert a collection of weak learners into a strong learner.

Where a strong learner is any algorithm that can be tuned to achieve arbitrarily 
good performance for some supervised learning problem.

### How is this accomplished?
By iteratively learning a set of weak models on subsets of the data you have at 
hand, and weighting each of their predictions according to each weak leaner's
performance.

You then combine all of the weak learner's predictions multiplied by their 
weights to obtain a single final weighted prediction that is much better than
any of the individual predictions themselves.

I'ts kind of incredible that this works as well as it does.

Here is a very basic example of boosting using 2 decision trees:

![](Pasted%20image%2020240603055101.png)

This example comes from the XGBoost documentation and shows that given a specific 
example, each tree gives a different prediction score depending on the data it sees. The 
prediction scores for each possibility are summed across trees, and the prediction is 
simply the sum of the scores across both trees.

Here, you can see that whatever it was we were trying to predict, the little boy had a higher 
predicted score summed across both trees than the old man.

Since we will be working with XGBoost's learning API for model evaluation, next,
it's a good idea to briefly provide you with an example that shows how model 
evaluation using cross-validation works with XGBoost's learning API (which is 
different from the scikit learn compatible API) as it has its own cross-validation 
capabilities baked in.

As a refresher, cross-validation is a robust method for estimating the expected 
performance of a machine learning model on unseen data by generating many 
non-overlapping train/test on your training data and reporting the average test 
set performance across all data splits.

[Measuring how good is your model](../supervised_learning_with_scikit-learn/03_Cross_Validation/03_Cross_Validation.md)

### Cross Validation Example for XGBoost

```python 

import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")

# Convert our dataset into an optimized data structure that the creators
# of XGBoost made that gives the package its lauded performance and effiency 
# gains called "DMatrix".
# In the previous exercise, the input datasets were converted into DMatrix data
# on the fly but when we use the XGBoost's Cross-Validation object, which is
# part of XGBoost's learning api we have to first explicitly convert our data 
# into a DMatrix.
churn_dmatrix = xgb.DMatrix(
                    data = churn_data.iloc[:, :-1],
                    label = churn_data.month_5_still_here
                )

# Creating a parameter dictionary to pass into our cross-validation. This is 
# necessary because the Cross Validation method has no idea what kind of 
# XGBoost model we are using and expects us to provide that information as 
# a dictionary of appropiate key-value pairs.
# In our parameter dictionary we are only providing the objective function
# we would like to use and the maximum depth that every tree can grow to.
params = {
    "objective": "binary:logistic", 
    "max_depth": 4
}

# Here we are calling the cross validation function, passing our DMatrix object
# storing all of our data, the parameter dictionary, the number of cross-validation
# folds, how many trees we want to build, what metric we want to compute, and
# wheter we want our output to be stored as a pandas dataframe.
cv_results = xgb.cv(
    dtrain=churn_matrix, 
    params=params,
    nfold=4,
    num_boost_round=10,
    metrics="error",
    as_pandas=True
)
# Convert our metrics into an accuracy and print the result
print("Accuracy: %f" % ((1-cv_results["test-error-mean"]).iloc[-1])
```

### Exercise
[Mesauring how good your model with XGBoost](../../extreme_gradient_boosting_with_xgboost/03_measuring_accuracy/xgb_measuring_accuracy.py)