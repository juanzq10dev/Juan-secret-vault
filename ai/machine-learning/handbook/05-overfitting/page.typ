= Overfitting
*What is it?:* Overfitting happens when a model learns the training data too well, including noise and random patterns, so it performs well on training data but badly on new (test) data.

*Problem:* A model that overfits doesn't generalize, so it can't handle unseen data.

*How to solve it:*
There are many techniques:
+ Use fewer features.
+ Apply regularization.
+ More training data.

== Regularization
*What is it?:* Regularization adds a penalty to the cost function to stop the model from becoming too complex and overfitting.

*Problem is solved*: Overfitting.

*How it works:*
We add penalization to all features (because some datasets have to many features):

$
  lambda / (2m) sum^n_( j=1 ) w^2_j
$

Where:
- $lambda$: regulation term $(lambda > 0)$ (it is an hyperparameter)
  - Big $lambda$ shall make the function underfit.
  - Small $lambda$ shall make the function still overfit.

=== Regularized linear regression
Implementing regularization in cost function:
$
  J(arrow(w), b) = 1 / (2m) sum^m_(i=1) ( f_(arrow(w), b) ( arrow(x)^((i)) ) - y^(( i )) )^2 + lambda / (2m) sum^n_( j=1 ) w^2_j
$

where:
- $lambda$: regulation term $(lambda > 0)$ (it is an hyperparameter)
  - Big $lambda$ shall make the function underfit.
  - Small $lambda$ shall make the function still overfit.

We need to implement regularization to gradient descent.
$
  text("repeat:") {
$
$
  w_j = w_j - alpha * partial / (partial w) J (arrow(w), b)
$
$
  b = b - alpha * partial / (partial b) J (arrow(w), b)
$
$
  }
$

Now with the regularized cost function gradient descent is:
$
  text("repeat:") {
$
$
  w_j = w_j - alpha * [ 1 / m sum^m_(i = 1) [ f_(arrow(w), b) (arrow(x))^((i)) - y^((i)) ]* x_j^((i)) + lambda / m w_j ]
$
$
  b = b - alpha * 1 / m sum^m_(i = 1) (f_((arrow(w), b) (arrow(x))^((i)) - y^((i)))
$
$
  }
$

=== Regularized logistic regression
We apply the regularization term to logistic cost function.

// $
// J = - 1 / m sum^m_i=1 [y^(i) log (f_(arrow(w), b) (arrow(x)^((i))) + (1 - y^((i))) log(1 - f_(arrow(w), b) (arrow(x)^((i))))]
// display(+ lambda / (2m) sum^n_(j = 1) w^2_j)
// $

#image("images/logistic-cost-function-regularized.png")

We apply the same to logistic gradient descent, it looks the same, but remember $f_(arrow(w), b) = g(z)$
$
  text("repeat:") {
$
$
  w_j = w_j - alpha * [ 1 / m sum^m_(i = 1) [ f_(arrow(w), b) (arrow(x))^((i)) - y^((i)) ]* x_j^((i)) + lambda / m w_j ]
$
$
  b = b - alpha * 1 / m sum^m_(i = 1) (f_((arrow(w), b) (arrow(x))^((i)) - y^((i)))
$
$
  }
$
