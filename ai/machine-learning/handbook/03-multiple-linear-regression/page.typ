= Multiple linear regression
== Training set terminology
#figure(caption: "Training set with multiple variables")[
  #image("images/multiple-linear-regression-dataset.png")
]

Now the training set has multiple variables, so there is new terminology:

- $x_j$ = $j^(t h)$ feature
- $n$ = number of features
- $arrow(x)^((i))$ = features of $i^(t h)$ training example
- $x^((i))_j$ = value of feature $j$ in $i^(t h)$ training example. (e. g. for this dataset $x^((4))_1 = 852$ )

== Linear regression with multiple variables
- *Main concept:* It is a method to predict an output based on two or more input variables
- *Problem solved:* Help to predict an outcome based on many factor. It also shows how important each feature is for prediction.
- *How it works:*
1. The model learns a formula $f$.
2. The computer finds best weights and bias that makes the formula fit the data.
3. Once trained, use the formula to predict new outcomes.

In the model $f$ for multiple linear regression there each feature $x$ has a weight $w$:

$
  f_(w, b) (x) = w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + ... + b
$

We can create fold values for $x$ and $w$ in vectors. So out model now looks like this:
$
  f_(arrow(w), b) arrow(x)= arrow(w) dot arrow(x) + b
$

Where:
- $arrow(x) = [x_1, x_2, x_n]$: vector of features.
- $arrow(w) = [w_1, w_2, w_n]$: vector of weights.
- $dot$ : dot product between vectors.

== Cost function
Cost function changes slightly, because it adapts to the vectors
$
  J(arrow(w) , b) = 1 / (2m) sum^(m)_(i = 1)(f_(w, b) (arrow(x)^((i)))- y^((i)))
$

== Gradient descent
Gradient descent formulas also has to adapt to the vectors. We have to calculate gradient descent for each $w$

$
  text("Repeat until convergence:") {
$
$
  w_1 = w_1 - alpha * 1 / m sum^m_(i = 1) (f_(arrow(w), b) (arrow(x))^((i)) - y^((i))) * x_1^((i))
$
$
  ...
$
$
  w_n = w_n - alpha * 1 / m sum^m_(i = 1) (f_(arrow(w), b) (arrow(x))^((i)) - y^((i))) * x_n^((i))
$
$
  b = b - alpha * 1 / m sum^m_(i = 1) (f_((arrow(w), b) (arrow(x))^((i)) - y^((i)))
$
$
  }
$

== Normal equation
Normal equation is an alternative to gradient descent, but:
- Only works for linear regression.
- It is slow when number of featurs is large $(> 1000)$

== Feature scaling
- *Main concept:* Is a way to adjust the range of features (input data) so they are on a similar scale.

- *Problem solved:*
  - Some algorithms (like gradient descent) work poorly if features have very different ranges (e.g age in years vs salary in dollars.)
  - Without scaling, features with large values dominate the learning process.
  - Scaling makes training faster and improves accuracy.

- *How it works*: We use different ways like mean normalization.

  Mean normalization scales values in a range between -1 and 1.
  $
    x_j = (x_j - mu_j) / (max(x_j) - min(x_j))
  $
  Where:
  - $x_j$: Is the feature we want to scale.
  - $mu_j$: Is the mean of the features.

  There are other ways like z-score.

== Feature engineering.
- *Main concept:* Is the process of creating, modifying or selecting feature to help a machine learning model to perform better.
- *Problem solved:*
  - Raw data is not often in the best form for a model.
  - Some features might be irrelevant for what we want to predict.
  - Some information might be hidden or not available.

- *How it works*
There many ways to achieve feature engineering:

1. Creating new features. (e.g from a "date" column, create day, month, year)
2. Selecting useful features.
3. Handle missing value (fill missing data with mean/median or other indicators)
