= Testing Machine Learning models

This section talks about some techniques we can use to test/debug a model.

== Testing linear regression

*How can we do it?*
+ Split dataset in two parts: e.g. 70% to train the model and 30% to test
+ Train the model with the test set. (You will get $arrow(w)$ and $b$)
+ Get cost function of the test set ($J_("test")$).
+ If $J_("test")$ is too high, model is not good. (Probably we have an overfit problem)

== Testing classification

*How can we do it?*

+ It is pretty similar to testing for linear regression.
+ But we will calculate when predictions are not good $hat(y) != y$

== Bias and variance
First, we need to know how to select a model.

=== Selecting a polynomial model
This is a technique to select how many polynomial degrees we can add.

*How can we do it?*
+ Split data in three: 60% to train, 20% cross validation (cv), 20% test.
+ Get cost function for all polynomials.
+ The one that gets the less cost error for cv is the better.

#figure(caption: "Selecting a good polynomial")[
  #image("images/selecting-polynomial.png")
]

*Note:* We can also use polynomials to correct neural networks too. So the same approach applies.

=== Diagnosing bias and variance.
We use this in order to know if we have high bias (underfit)or high variance problem (overfit)

*How can we do it?*
+ Split data in three: 60% to train, 20% cross validation (cv), 20% test.
+ Get cost function for training set $J_("train")$ and cost function of cv $J_("cv")$.
+ Then:
  - If $J_("train") approx J_("cv")$ and $J_("train")$ is high: we have high bias (underfit).
  - If $J_("cv") >> J_("train")$: we have high variance (overfit).
  - If $J_("cv") >> J_("train")$: and $J_("train")$ we have high variance and high bias.

  #figure(caption: "High vias and high variance based on degree of the polynomial")[
    #image("images/bias-and-variance-base-on-polynomial.png")
  ]

=== Choose a good regularization term $(lambda)$
We use this to choose which value is good to regularization term ($lambda$) of cost function.

*How can we do it?*
+ Split dataset in three sets: training, cross validation (cv) and test.
+ Compute cross validation error for all of them (use cost function with regularization).
+ The smaller value is the regularization term that is correct.

=== Establish a level of performance
Humans also have errors, we can base on that to select a reasonable level error of performance and diagnose bias and variance problems based on that.

*How can we do it?*
+ Choose a baseline performance based on human error.
+ Get the training set cost function ($J_("train")$).
+ Get the cross validation cost function ($J_("cv")$).
+ If:
  - $"Baseline performance" > J_("train")$: Means high bias.
  - $J_("train") > J_("cv")$: Means high variance.
  - We can also have both.

=== Adding more data:
- Adding more data does not help when we have high bias.
#figure(caption: "High bias")[
  #image("images/high-bias.png")
]

- Adding more data helps when we have high variance:
#figure(caption: "High variance fixed adding more data")[
  #image("images/high-variance.png")
]

=== Summary
Those are all debugging solutions for a model in summary
#figure(caption: "Summary of solutions")[
  #image("images/debugging-solutions.png")
]

== Bias/variance and neural networks
There are some things we need to know about neural networks:

-  Neural networks fix bias and variance this way:

#figure(caption: "Bias and variance in neural network")[
  #image("images/neural-network-cycle.png")
]

- Neural networks can use regularization term too ($lambda$).