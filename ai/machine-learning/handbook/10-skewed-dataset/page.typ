= Working with Skewed Dataset
A skewed dataset is and imbalanced dataset where one class has way more examples that other (e.g a dataset of with rare disease that only happens once)

== Precision and recall
*What they are?* Other metrics we use when working with skewed datasets.

*Problem they solve:* When working with skewed datasets standard average is misleading.

*How can we get them:*
+ We use a confusion matrix. Example:
  #figure(caption: "Confusion matrix")[
    #image("images/confusion-matrix.png", height: 40%)
  ]
+ We apply a formula to get them. (explained in next sections)

=== Precision
*What is it?* A metric that tells how many positives predictions are actually correct.

*How do we get them?*
+ We use a confusion matrix.
+ Using results of the confusion matrix we can calculate it:
  $
    "True positives" / "Predicted positives" = "True positives" / ("True positives" + "False positives.")
  $

=== Recall
*What is it?* A metric that tells how many actual positives were actually detected by the model.

*How does it work?*
+ We use a confusion matrix.
+ Using results of the confusion matrix we can calculate it:
  $
    "True positives" / "Actual positives" = "True positives" / ("True positives" + "False negatives.")
  $

=== Precision vs Recall
*Key point:* Usually having a high precision reduces recall and vice versa. Based on the problem we want to resolve we need to select a good threshold.

*How does it works?*
- There is relation between recall, precision and threshold.
  - If we increase threshold (make predictions more confident: precisions increases, but threshold decreases.
  - If we decrease threshold (avoid missing too many cases): precision decreases, but threshold increases.

  #figure(caption: "Relationship between recall vs precision and threshold")[
    #image("images/recall-vs-precision.png")
  ]

=== F1 Score
*What is it?* An algorithm to automatically get a good trade off between precision of recall.

*Problem it solves:* It helps to get a good trade off between precision and recall automatically. (Usually you would choose that manually)

*How can we use it?*
When we have models with different thresholds:
+ We calculate precision and recall for each of them:
+ We get $F_1$ with the following formula:
  $
    F_1"score" = 1 / (1/2 (1/"precision" + 1/"recall")) = 2 ("precision" * "recall") / ("precision" + "recall")
  $
+ The model with the more balanced $F_1 "score"$ (closed to 0.5) is the best.
