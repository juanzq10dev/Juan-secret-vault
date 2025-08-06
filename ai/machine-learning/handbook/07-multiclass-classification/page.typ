= Multiclass classification

*Definition:* This is a type of machine learning problem where the model must choose one label from more than two possible classes.

*Problem it solves:*
- Problems where we want to classify between multiple categories (e.g. classify a picture as cat, dog or rabbit).
- It is a generalization of logistic regression.

*How it works:*
#figure(caption: "Neural network with softmax output")[
  #image("image/softmax-in-neural-networks.png")
]
+ The model takes input features.
+ It processes data through layers in the neural network.
+ Output layer has a neuron for each class.
  $
    z_j = arrow(w_j) * x + b_j
  $
+ Softmax activation is applied to get probabilities for each class:
  $
    a(z) = e^(z_j) / (sum_(k=1)^N e^(z_k)) = P (y = j|arrow(x))
  $

  This is the softmax activation function, where::
  - $z_j$: The $z$ value for $j_(t h)$ class.
  - $N$: The total number of classes.
  - $j$: The current class $j = (1, 2, ..., N)$
  - $P (y = j|arrow(x))$: Probability of $arrow(x)$ to belong to class $j$
+ The class with the highest probability is the final prediction

== Loss function (Cross entropy loss function)
Loss function for softmax regression is:
$
  l o s s (a_1, ..., a_y, a_N) = cases(
    -log(a_1) "if" y = 1,
    -log(a_2) "if" y = 2,
    ...,
    -log(a_N) "if" y = N
  )
$

That can be defined as:
$
  l o s s (a_y) = cases(-log(a_j) "if" y = j)
$
