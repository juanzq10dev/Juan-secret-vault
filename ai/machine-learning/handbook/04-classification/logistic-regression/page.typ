== Logistic regression

- *What is it? * It is a classification algorithm used to predict probabilities for two or more classes.

- *Problem solved:* Predicts probability between 0 and 1 for a feature to belong to a class.

- *How it works*
  + It calculates linear regression on a function $z$.

    $
      z = arrow(w) dot arrow(x) + b
    $
  + It passes the linear regression $z$ to the sigmoid function $g(z)$.
    $
      g(z) = 1 / (1 + e^(-z))
    $

  + Result of $g(z)$ is the probability that input belongs to a class or not depending on decision boundary and predefined threshold.

  #figure(caption: "Logistic regression")[
    #image("images/logistic-regression.png")
  ]

=== Decision boundary
- *What is it?:* An imaginary line or surface that a classification model uses to separate different classes in the feature space.

- *Problem solved:* Helps the model decide which class a new data point belongs to.

- *How it works:*
  + Decision boundary is the result of solving $z$ when $z = 0$:
  $
    z = arrow(w) dot arrow(x) + b
  $
  $
    0 = arrow(w) dot arrow(x) + b
  $
  $
    arrow(w) dot arrow(x) = b
  $
  + For logistic regression this produces a straight line.

=== Cost function (Squared Error Cost)


Cost function for logistic regression is different:

$
  J(arrow(w), b) = 1 / m sum^m_(i = 1) L(f_(arrow(w), b) (arrow(x)^((i))), y^((i)))
$

Where $L(f_(arrow(w), b) (arrow(x)^((i))), y^((i)))$ (we will just name it $L$) is named the "loss function" and is:
$
  L = cases(
    - log(f_(arrow(w), b) (arrow(x)^((i)))) "if" y^((i)) = 1,
    - log(1 - f_(arrow(w), b) (arrow(x)^((i)))) "if" y^((i)) = 0
  )
$

This can be represented as:

#image("images/loss-function.png")


So, applying all function and taking away the negative sign complete loss function is:
$
  J = - 1 / m sum^m_i=1 [y^(i) log (f_(arrow(w), b) (arrow(x)^((i))) + (1 - y^((i))) log(1 - f_(arrow(w), b) (arrow(x)^((i))))]
$

=== Gradient descent
Gradient descent for logistic regression is:
$
  "repeat" {
$
$
  w_j = w_j - alpha partial / (partial w_j) J (arrow(w), b)
$
$
  b = b - alpha partial / (partial b) J (arrow(w), b)
$
$
  }
$

Applying cost function and solving the partial derivatives it is:
$
  "repeat" {
$
$
  w_j = w_j - alpha [ 1 / m sum^m_(i = 1) ( f_(arrow(w), b) (arrow(x)^((i))) - y^((i))) x^(i)_j]
$
$
  b = b - alpha [ 1 / m sum^m_(i = 1) ( f_(arrow(w), b) (arrow(x)^((i))) - y^((i))) ]
$
$
  }
$

But remember:
- $f_(arrow(w), b) (arrow(x)^((i))) = g(z) = 1 / (1 + e^(z))$
