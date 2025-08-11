= Machine Learning Development Process

As in Software Development, Machine Learning apps also have a development process:

#figure(caption: "Machine Learning development process")[
  #image("image/ml-dev-process.png")
]

== Error Analysis

*What is it?* A process to diagnose and fix errors on machine learning models.

*Problem it solves:* We need a way to increase performance of our model when it makes too much failures on predictions.

*How can we do it?*
This is example:
+ We manually examine examples that have failed and categorize them.
+ We add more data features of the categories we want to improve

== Adding data to model
=== Data augmentation
*What is it?* It is the process to add more data based on modifying existing example.

*Problem it solves:*
- Useful when there is not much data.
- To reinforce model training.

*How we can do it?* By adding noise or distortions to our examples. e.g: rotating an image, increasing size of an image, adding background noise to an audio.

=== Data synthesis
*What is it?* It is the process of generating synthetic data (artificial data that mimics real world data)

*Problem it solves:* Sometimes it gets hard to get real world data.

*How can we do it?* There are different synthesis methods e.g: generating fake profiles, artificial bank transactions and so on:

== Transfer learning
*What is it?* It is when we use data from other neural network on ours.

*Problem it solves:*
- Sometimes we cannot gather enough data.
- Training big models require lot of computations.
- It may save time, resource, and increase performance.

*Why does this work?* Because early hidden layers learn to detect generic features, that may work for different tasks.

*How can we do it?*
+ Get an already trained model with similar input to the model you want to do. (You can download it, or pre trained it).
+ You have two options:
  + Replace output layers by the ones you want.
  + Retrain all parameters by adding examples of your dataset (This is fine tuning.)

== Full cycle of a machine learning project
This is the full cycle of an machine learning project, it is pretty similar to a software development project.

#figure(caption: "Full cycle of a machine learning project")[
  #image("image/ml-app-full-cycle.png")
]

=== Fairness, bias, and ethics.
*What is it?* Some kind of recommendations that we must follow in order to make safe systems.

*Problem it solves:*
Sometimes, bias may learn harmful patterns (like get racists, or discriminate women).

*How can we do it?*
- Audit systems against possible harm after deployment.
- Be careful with the data you use to train the model.
- Try to add a variety of data.
