= Reinforcement learning
== Reinforcement learning
*What is it?*\
A branch of machine learning where an agent learns to make decisions based on a trial-an-error process.

*Terminology*\
#table(
  columns: 3,
  [Symbol], [Concept], [Explanation],

  [$s$],
  [State],
  [
    - A representation of the environment at a particular point in time.
    - $s'$ stands for new state
  ],

  [$a$], [Action], [- A decision or move the agent can make.],
  [$R, R(s), R_1$],
  [Reward],
  [
    - A scalar feedback received after taking an action.
      - $R(s)$: Reward in state $s$,
      - $R_1$: Reward in step 1.
  ],

  [$gamma$],
  [Discount factor],
  [
    - An hyperparameter that determines value of future rewards.
    - $gamma = 0$ makes the agent myopic (only immediate rewards matter), while $gamma arrow.r 1$ makes future rewards more important.
    - Usually $gamma$ is near to 1, but never 1 (0.9, 0.9999)
  ],

  [$pi$], [Policy], [- A map of states that tells the agent which action to take on current state. $pi(s) = a$],
  [$Q(s, a)$],
  [State action value function],
  [Return if:
    + Start in state $s$.
    + Take action $a$ (once).
    + Behave optimally
  ],

  [-], [Return], [- Ponderation between reward and state discounted by time step],
  [-], [Terminal state], [- A state where episode ends, reward is received and nothing else happens],
)

*Important formulas*\
#table(
  columns: (0.5fr, 0.75fr, 2fr),
  [Concept], [Explanation], [Formula],

  [Return],
  [Calculate the return],
  [
    $ "Return" = R_1 + gamma R_2 + gamma^2 R_2 + ... "until terminal state" $],

  [Bellman equation],
  [A formula to compute state value equation ($Q$)],
  [
    $
      Q(s,a) = R(s) + gamma max_a' (s', a') \
      Q(s,a) = R_1 + gamma max [R_2 + gamma R_3 + gamma^2 R_4]
    $
  ],
)

*How it works?* \
The goal is to find a policy $pi$ that tells what action ($a = pi(s)$) to take in every state $s$ to maximize the return.
+ The Agent looks at the current state $s$.
+ Based on its policy, the agent performs and action $a$.
+ The environment changes to a new state $s'$.
+ The agent receives a reward $R(s)$ and observes the new state $s'$.
+ The agent uses Bellman Equation $Q(s, a)$ to adjust its guess.

=== Stochastic (random) environment
- In some applications the outcome is not completely reliable, so it is a bit random.
- In that case we calculate the expected return using the average:
$
  Q(s,a) = R(s) + gamma E [max_(a') Q(s', a')]
$
Where:
- $E$: is the average.

== Deep reinforcement learning
- Reinforcement learning with neural networks.

=== Terminology
#table(
  columns: 3,
  [Symbol], [Concept], [Explanation],

  [
    $
      s = mat(x; y; gamma)
    $
  ],
  [Continuous state space],
  [
    - A space that depends on more than one values:
      - Example: A helicopter may have as state: position in $x$, in $y$, inclination $gamma$, etc.
  ],
)

=== How it works:
- We use neural networks that takes the state $s$ and a set of possible actions $a$.
- In this example we have four possible actions:
#figure(caption: "Deep reinforcement learning")[
  #image("images/reinforcement-learning.png")
]

- In the previous example, we had to train for each action $a$. This is an optimized architecture, that calculate all actions at the same time:
#figure(caption: "Better architecture")[
  #image("images/better-architecture.png")
]
- The main goal is to action $a$ that maximizes $Q(s, a)$

== Algorithm refinement
=== $epsilon$ greedy policy
*What is it?* \
An hyperparameter that represents a probability to pick an action $a$ randomly:

*Problem it solves:*\
- If the algorithm a always chooses the better path learned, it will never choose a possible better solution.

*How it works:* \
It is based on Exploitation and Exploration:
- Exploitation: With probability $1 - epsilon$, pick the action that maximizes $Q(s, a)$.
- Exploration: with probability $epsilon$, pick an action $a$ random}.

A common strategy is to start with $epsilon$ high and then decrease it.

=== Mini batches
*What it is?* \
A technique to manage large datasets, works in both supervised learning and reinforcement learning

*Problem it solves*\
- Cost function and gradient descent take too much on large datasets.

*How it works* \
+ We pick a subset of params ($m'$).
+ In each iteration we use different $m'$ params to create a mini batch.
+ Instead of using all the examples we put a mini batch to train the model.

=== Soft update
*What it is?*\
- A technique that uses hyperparameters to make more gradual change of $Q$, or to the neural networks params $w$ and $b$.

*Problem it solves*\
- Helps to prevent worse values for the new $Q$, or to $w$ and $b$.

*How it works?*\
We use to hyperparams for both $w$ and $b$:
$
  w = 0.01 w_("new") + 0.99 w \
  b = 0.01 b_("new") + 0.99 b
$
