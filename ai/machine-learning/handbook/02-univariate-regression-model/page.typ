= Linear Regression Model (Univariate)

Before learning linear regression we need to know how to interpret a training set.
== Training set terminology
#figure(caption: "How a training set looks like.")[
  #image("images/training-set.png", height: 25%)
]
- $x$: feature (the input variable)
- $y$: target (the output)
- $m$: number of training examples.
- $(x, y)$: single training examples.
- $(x^((i)), y^((i)))$: $i^(t h)$ training example

== Linear regression with one variable

- *Main concept:* Linear regression model fits a straight line to dataset in order to predict new values.
- *Problem solved:* It predicts numerical values based on other information (e.g the note of an student based on how much time it studies).
- *How it works:*
  + Train the learning algorithm with a labeled dataset that includes both, the inputs (features) and the correct outputs (targets)
  + The algorithm creates a function $f$ called *model* (or hypothesis)
  + The model $f$ takes a new input $x$ and gives a prediction $hat(y)$.

  In linear regression model $f$ is represented by:
  $
    f_(w, b)(x) = w x + b
  $

  Where:
  - $x$ is an input of $f$.
  - $w$ is the slope (how much $f$ changes when $x$ changes.
  - $b$ is the starting point (when $x$ is 0).
  - $f_(w, b)(x) = hat(y)$ where $hat(y)$ is the prediction.

We want to get values for $w$ and $b$ that make the model $f$ fit well out dataset so that prediction $hat(y)$ is closer to real value $y$ for many examples.

== Cost function (Squared error cost function)

- *Main concept:*
  - Cost function quantifies the error between predicted values from the model $f$ and actual values $y$ from the dataset.
  - It provides a single value that algorithm tries to minimize during training.

- *Problem solved:*
  - Tells us how well the model performs.
  - Helps to fine the best-fit line by minimizing the difference between predicted outputs and actual output.

- *How it works:*
  Cost functions model is a formula like this:
  $
    J(w, b) = 1 / (2m) sum^(m)_(i = 1)(hat(y)^((i)) - y^((i)))
  $

  Like $f_(w, b)(x) = hat(y)$, it can also be represented like:
  $
    J(w, b) = 1 / (2m) sum^(m)_(i = 1)(f_(w, b) (x^((i)))- y^((i)))
  $

  Where:
  - $m$ is the number of training examples.

  The main objective is to minimize the result of the cost function.

  In linear regression cost function will always be a convex function.
== Gradient descent

- *Main concept:*
  Gradient descent is a method computers use to find the best fit for linear regression by constantly improving the parameters.

- *Problem solved:*
  It helps the model to find the best fit for parameters $w$ and $b$.

- *How it works:*
  It's like walking down a hill step by step until you reach the lowest point:
  + Starts with a random values for $w$ and $b$ (usually 0)
  + Use the cost function to calculate the error.
  + Calculate the better gradient (line that reduces the error the fastest).
  + Advance to the better gradient.
  + Repeat until convergence (until the error is as small as possible.)

  To achieve the optimal values for $w$ and $b$ use the following formula:

  $
    text("Repeat until convergence:") {
  $
  $
    w = w - alpha * partial / (partial w) J (w, b)
  $
  $
    b  = b - alpha * partial / (partial b) J (w, b)
  $
  $
    }
  $

  Solving the partial derivate for cost function:
  $
    text("Repeat until convergence:") {
  $
  $
    w = w - alpha * 1 / m sum^m_(i = 1) (f_(w, b) (x)^((i)) - y ^((i))) * x ^ ((i))
  $
  $
    b  = b - alpha * 1 / m sum^m_(i = 1) (f_(w, b) (x)^((i)) - y^((i)))
  $
  $
    }
  $

  Where:
  - $alpha$ its called the learning rate, it's how big the steps are when the algorithm moves toward the minimum error (it is an hyperparameter #footnote("An hyperparameter is a variable you set manually").
    - Too big learning rate (like 1) may cause the algorithm to diverge (never converge).
    - Too small learning rate (like 0.0001) will make the algorithm to take too many steps to reach the best solution.
  - $m$ is the number of training examples.

  *Note:* Remember $f_(w, b)(x) = hat(y)$, so you can replace that on the formula too.

  *Note:* Gradient descent stops when a local minima is found. But in linear regression there is only one local minima because the function is convex. 
