= Anomaly Detection

*Definition:* It is an algorithm to detect anomalies.

*Problem it solves:* It learns to detect unusual events.

*Examples:*
- Detect fraud accounts.
- Detect failed manufacturing.

*How it works:*
+ We train the algorithm with a dataset full of features without anomalies.
+ The algorithm will learn to detect patterns.
+ When a new example is passed the algorithm is able to detect unused patterns and tells us the probability of feature $x$ to be seen in the dataset.

== Gaussian (Normal) Distribution
*Definition:* It is a probability distribution that describes how data points are spread around the mean.

*Problem it solves:*
- It helps us to know the probability of one value to appear.

*How it works:*
- It is calculated by this formula:
$
  p(x) = 1 / (sqrt(2 pi) * sigma) * e^(-(x - mu)^2 / 2sigma^2)
$

$
  mu = 1 / m sum_(i = 1)^m x^((i))
$

$
  sigma^2 = 1 / m sum_(i = 1)^m (x^((i)) - mu )^2
$

It looks like a bell:

#figure(caption: "Gaussian distribution")[
  #image("images/gaussian_distribution.png")
]
Where:
- $p(x)$: probability of x to appear in dataset.
- $mu:$ mean.
- $sigma$: standard deviation (spread of the curve).
- $m$: number of elements.

== Density estimation
*Definition:* It is a technique that tells us how likely are different values to occur.

*Problem it solves:*
- It is the Anomaly Detection Algorithm.
- It helps us to detect unusual values.

*How it works:*
We use normal distribution to calculate probability of one value. We just multiply probability of all features of an example:

+ Choose $n$ features from $x_i$ that you think might be indicative of anomalous examples.
+ Fit parameter $mu$ and $sigma^2$ for each feature.
  $
    mu_j = 1 / m sum_(i = 1)^m x_j^((i))
  $

  $
    sigma^2 = 1 / m sum_(i = 1)^m (x_j^((i)) - u_j)
  $
  Where:
  - $j$: number of feature.
  - $i$: number of value.
  - $x^(i)_j$: feature $j$ of the $i^(t h)$ value.
+ Given new example $x$, compute $p(x)$:

  (Notice this is just product of normal distribution of all features $x$)
  $
    p(x) = product (p(x_j, mu_j, sigma^2_j)) = product^n_(j = 1) 1 / sqrt(2pi) sigma_j * e^(-(x_j - mu_j)^2 / (2 sigma^2_j) )
  $
  Where:
  - $product$: means product of all.
+ Compare if $p(x) < epsilon$, where $epsilon$ is a defined threshold.

== Real number evaluation
*Definition:* It is a way to evaluate how good is our algorithm working.

*Problem it solves:* We need a real number that tells us how good is our model doing.

*How it works:*
+ Split dataset in three:
  - Training set: full of normal values (non-anomaly).
  - Cross validation: a bit of anomalous examples, but majority of normal values.
  - Test set: a bit of anomalous examples, but majority of normal values.
+ Train algorithm with training set.
+ Use cross validation set to tune $epsilon.$
+ On test example:
  $
    y = cases(
      1 "if" p(x) < epsilon ("anomaly"),
      0 "if" p(x) >= epsilon ("normal")
    )
  $

*Recommendation / Considerations:*
- If there is too little data we might not have a test set, in that case use only cross validation set.
- Many concepts learn for skewed-dataset section, can be apply:
  - Precision / Recall.
  - Selecting many $epsilon$, getting $F_1$, and the major $F_1$ wins.

== Anomaly Detection vs Supervised Learning.

*Anomaly Detection:*
- Useful when we have very small positive examples.
- If there are many types of anomalies, or future anomalies make look very different (because it used unlabeled data).

*Supervised Learning:*
- Large number of positive and negative examples (because we need labeled data.)
- If future anomalies will look like the existing.

== Recommendations / Tips

=== Choosing which feature to use.
- Because we have unlabeled data, it is important to choose good features.
- Better choose gaussian features.

=== How to deal with no Gaussian features
- Try applying a function to all data that may look data like a Gaussian distribution (e.g $log(x), x^2, sqrt(x), x^2 + 1, ...$)

#figure(caption: "Transforming features to gaussian")[
  #image("images/non_gaussian_distribution.png")
]

=== Error analysis
We may have a huge $p(x)$ for anomalies an non anomalies values, this will make the algorithm to fail.

#figure(caption: "An anomaly that looks like normal")[
  #image("images/error-on-analysis.png")
]

In that case find sus features and add them to the algorithm.

