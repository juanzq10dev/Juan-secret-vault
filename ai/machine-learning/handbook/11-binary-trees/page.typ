= Decision Trees

== Decision Trees
*Main concept:* Decision Trees are a supervised learning model that splits data into branches based on feature values to make prediction.

*Problem they solve:* Helps classify or predict outcomes by breaking down complex decisions into simple rules.

*How it works:*
- Start with all data at root node.
- Choose the best feature to split on at decision nodes, using metrics like *information gain*.
- Keep splitting until you reach leave nodes (final decisions.).

#figure(caption: "Decision tree example")[
  #image("images/decision-trees.png")
]

== Choosing a split feature
+ Make different splits using for each feature.
+ For each split calculate the information gain.
+ The one with more information gain is the better.

All concepts are explained in the next section.

=== Purity
*Main concept:* A measure of how uniform a dataset is. (how many labels of the same class it has)

*Problem it solves:* Used to decide the quality of a split by maximizing purity.

*How it works:* A node is pure if it contains only one class. Higher purity means the node is better for prediction.

=== Entropy
*Main concept:* A measure of disorder or uncertainty in the data. (More entropy mean less pure)

*Problem it solves:* It measures impurity of a dataset.

*How it works:* It uses this formula:
$
  H(p_1) = - p_1 log_2(p_1) - p_0 log_2(p_0)
  H(p_1) = - p_1 log_2(p_1) - (1 - p_1) log_2(1 - p_1)
$

Where:
- $p_1$: percentage of examples belonging to positive class
  $ p_1 = "N° of examples on a side that belong to positive value" / "N° of examples on a side" $.
- $p_0$: probability of belonging to negative class $p_0 = 1 - p_1$
- H(p_1): Entropy

=== Information gain
*Main concept:* The reduction in entropy after splitting data with a feature.

*Problem it solves:* Decides which feature is best for splitting in a decision tree.

*How it works:*
+ Calculate entropy before the split and after.
+ The bigger the drop, the higher the information gain.

$
  H(p_(1 "root")) - (w_("left") H (p_1 "left") + w_("right") H(p_1 "right")
$

Where:
- $H(p_(1 "root"))$: Entropy before the split.
- $w$: percentage of examples in a side of the split:
  $
    w_("side") = ("N° of a values on current side" / "N° of values")
  $

== Features with multiple values
To deal with features that have more than one possible value we use one-hot encoding

=== One-hot encoding
*Main concept:* A technique to turn categorical values (like "red", "blue") into numbers on a dataset.

*Problem it solves:* Machine learning models need numbers, not words.

*How it works:* Each category becomes a new column with 1 for presence and 0 for absence. For example

#table(
  columns: (1fr, 1fr),
  inset: 10pt,
  align: horizon,
  table.header([], [*Color*]),
  [Toyota], [Red],
  [Suzuki], [Blue],
  [Subaru], [Green],
)

#table(
  columns: (1fr, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header([], [*Red*], [*Blue*], [*Green*]),
  [Toyota], [1], [0], [0],
  [Suzuki], [0], [1], [0],
  [Subaru], [0], [0], [1],
)

== Continuous features
*Main concept:* Features that can take any value in a range (like height, weight, temperature).

*Problem it solves:* Many real-world data points are numbers, not categories, and need special handling in models.

*How to deal with them:* Split them by choosing a threshold (e.g., "Temperature ≤ 30°C?").

This is how to choose a threshold.

+ Select different values for the threshold.
+ As threshold splits data on two sides, you can calculate information gain.
+ Keep iterating until you find the highest information gain.

== Regression Trees
*Main concept:* A decision tree used for predicting numbers instead of classes.

*Problem it solves:* Helps predict continuous values using trees (like price or temperature).

*How it works:*
- Similar to decision trees: splits data into regions, and each leaf outputs the average value of that region.
- Variance is used instead of entropy. Reduction of variance is used instead of information gain (which actually the same approach as information gain, but using variance.)

== Random forest
*Main concept:* An ensemble of many decision trees combined together.

*Problem it solves:*
- Reduces overfitting and improves accuracy compared to a single tree.
- Trees are highly sensitive to small changes on a dataset.

*How it works:*
+ Each tree is trained on random samples (using sampling with replacement) and random features (select a random subset of features and let the algorithm choose from it).
+ Each tree makes a decision.
+ The final result is the majority vote if classification, or average if regression.

=== Sampling with replacement
*Main concept:* A way of picking random samples where the same item can be chosen more than once.

*Problem it solves:* Creates diverse training sets for ensemble methods.

*How it works:* Each time you pick an item, you put it back, so it can be picked again.

== XGBoost
*Main concept:* A very efficient and powerful boosting algorithm for decision trees, it is one of the most used.

*Problem it solves:* Improves accuracy and speed of decision trees.

*How it works:*
- Works like random forest, but when using sampling with replacement we make more likely to pick misclassified examples (examples that were categorized wrong) from previous trained trees.

== Decision trees vs Neural networks
Decision trees:
- Work well with tabular (structured) data.
- Not for unstructured data (like images).
- Faster training.
- More human readable.

Neural networks:
- Work well with all type of data.
- Slower training.
- Works with transfer learning.
