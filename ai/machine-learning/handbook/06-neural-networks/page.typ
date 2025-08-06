= Neural networks

#figure(caption: "Neural network")[
  #image("images/neural-network-example.png"),
]
*What is it?* A neural network is a type of machine learning inspired by the structure of human brain. It's made up of layers of simple computing units (neurons) that work together to learn patterns from data.

*Problem it solves:*
- Traditional models struggle (have low performance) with complex, nonlinear patterns in data.
- Neural networks can automatically learn features and handle very complicated tasks like image recognition, translation, videogames, and so on.
- They are used when you do not have a clear set of rules, but you do have a lot of examples.

*How it works:*
- Structure: A neural network has three layers.
  + Input layer: receives raw data.
  + Hidden layer: process data with weights and activation functions
  + Output layer: produces the final prediction.
- Forward propagation:
  + Data moves from input to hidden layers to output.
  + Each neuron multiplies input by a weight, adds a bias and applies activation function.
  + Each neuron output becomes the input of the next layer.
- Training:
  + The network makes a prediction.
  + The cost function measures the error.
  + Then uses backpropagation and gradient descent to update weights and reduce the error.

== Neural network layer
*What is it?*
Is a collection of neurons that process input data in parallel.

*Problem it solves:*
Layers allow the network to break down complex patterns into smaller, manageable transformations.

Each layer learns to represent the data in a slightly more complex way than the one before.

Example:
- First hidden layer might detect edges in an image
- Next one might detect shapes
- Later ones recognize objects

*How it works:*
+ Each layer is composed from neurons.
+ Neurons calculate values in parallel and pass the value to the next layer (This is called forward propagation).

== Neurons

*What is it?*
A neuron is the basic of a neural network. It receives input, processes it, and sends the result to the next layer.

It's inspired by biological brain cells, but it's a math function, not a real brain cell.

*Problem it solves*
- Helps the network process and learn patterns
- Each neuron contributes the overall prediction by focusing on small parts of input.

*How it works*
+ Inputs come from the previous layer
+ Calculates a weighted sum of all inputs plus bias.
  $
    z = arrow(w) * arrow(x) + b
  $
+ Applies an activation function (sigmoid#footnote("Sigmoid the the function from logistic regression"), or ReLU).
  $
    a = g(arrow(x))
  $
Where:
- $a$: is the result of the activation function, is called _activation_
- $g$: represents the activation function.


== Forward propagation

*What is it?*
Forward Propagation is the process where input data flows through the network, layer by layer, until it reaches the final output.

It's how a neural network makes predictions.

*Problem it solves*
- It allows the model to combine inputs, weights, and activations to make a decision.
- It's the first step in training or testing a neural network.

*How it works*
+ Input layer receives data
+ Each neuron in hidden layer provides an activation (output).
+ The output of one layer becomes the input for the next layer.
+ This continues through all hidden layers.
+ Final output is the model's prediction.


== Activation functions

*Definition*: Activation functions are non-linear functions applied to the neuron output.

*Problem it solves:* 
- Without them neural networks would just be linear regression.

*How it works:*
+ A neuron calculates a weighted sum.
+ The activation function takes that value and transforms it.
+ This activation value is passed to the next layer.

Note: Hidden layers usually use ReLU.

*Common activation functions*
#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header([Function], [Formula], [Use & Behavior]),
  "ReLU", 
  $
    f(z) = max(0, z)
  $,
  "Keeps only positive values",
  "Sigmoid",
  $
    f(z) = 1 / (1 + e^(-z))
  $,
  "Used in binary classification. Outputs between 0 and 1.",
  "Softmax",
  $
    f(z_i) = e^(z_i) / (sum (e^z_j)) )
  $,
  "Used in binary classification. Outputs between 0 and 1.",
)