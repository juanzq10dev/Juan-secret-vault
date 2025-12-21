= Recommender Systems

*Definition:* It is an AI technique that suggests items to users based on their preferences and behavior.

*Problem it solves:*
- They provide system personalize choices, improving user experience and engagement.

*How it works:*
There are two main approaches:

+ Collaborative filtering: Based on preferences of similar users.
+ Content based filtering: Based on similar items a user liked before.

== Collaborative filtering

*Definition:* Suggest items based on the preferences of other users.

*How it works:*
Example:

=== Intuition: recommendations using linear regression
Given a dataset of ratings, users and features, we can predict the rating of a movie doing linear regression.

#figure(caption: "Dataset with values for features")[
  #image("images/dataset-features.png")
]

$
  "user's (j) rating for movie (i)"= w^((j)) dot x^((i)) + b^((j))
$

We will use this cost function $J(w, b)$:
$
  1 / 2 * sum_(i:r(i, j) = 1) ( w^((j)) dot x^((i)) + b^((j)) - y^((i, j)) )^2 + lambda /2 sum_(k = 1)^n ( w_k^((j)) )^2
$

Where:
- $r(i, j)$: I is 1 if user has $j$ has rated movie $i$, 0 otherwise.
- $y^(i, j)$: rating given by user $j$ on movie $i$ (if defined).
- $w^((j)), b((j))$: parameters for user $j$.
- $x^((i))$: feature vector for movie $i$. example of features: romantic, action, fiction, fantasy (for this case we have a dataset with values for this features.)
- $m^((j))$: no. of movies rated by user $j$.
- $n$: no. of features.

Notice we also have added the regularization term, which may be there or not.

The above is just for one user. Now, if we want to get $w, j$ for all users, we use this $J( w^((1)), b^((1)); ..., w^((n_u)), b^((n_u)) )$:

$
  J( w^((1)), b^((1)); ..., w^((n_u)), b^((n_u)) ) = \
  1 /2 sum_(j = 1)^n_u sum_(i:r(i, j) = 1) ( w^((j)) dot x^((i)) + b^((j)) - y^((i, j)) )^2 + lambda / 2 sum_(j = 1)^n_u sum_(k = 1)^n ( w_k^((j)) )^2
$

where:
- $n_u$: is the number of users

=== Collaborative filtering algorithm
*Definition:* In this case we do not have values for features $x$.

#figure(caption: "Dataset with no values for features")[
  #image("images/dataset-no-features.png")
]

*How it works:*
We use cost function to minimize $x^((i))$ with the actual rating. Notice this is possible because we have ratings of other users.

For a single feature vector $x^((i))$:
$
  J( x^((i)) ) = 1 / 2 sum_(j:r(i,j) = 1) ( w^((j)) dot x^((i)) + b^((j)) - y^((i,j)) )^2 + lambda / 2 sum_(k = 1)^n (x_k^((i)))^2
$

For all feature vectors $x$
$
  J( x^((1)), ..., x^((n_m)) ) = \
  1/2 sum_(i = 1)^(n_m) sum_( j:r(i, j) = 1 ) ( w^((j)) dot x^((i)) + b^((j)) - y^((i, j)) )^2 + lambda / 2 sum_(i = 1)^n_m sum_(k = 1)^n (x_k^((i)))^2
$

Where:
- $n_m$ number of movies.
- $n:$ number of features.

Now to build collaborative filtering algorithm we put $J( x^((1)), ..., x^((n_m)) )$ and $J( w^((1)), b^((1)); ..., w^((n_u)), b^((n_u)) )$ together:

$
  J(w, b, x) = 1 / 2 sum_( (i,j):r(i,j) = 1 ) ( w^((j)) dot x^((i)) + b^((j)) - y^((i, j)) )^2 + \ lambda / 2 sum_(j = 1)^(n_u) sum_(k = 1)^n
  (w_k^((j)))^2 + lambda / 2 sum_(i = 1)^n_m sum_(k = 1)^n (x_k^((i)))^2
$

Gradient descent is pretty similar to linear regression gradient descent, but this time $x$ is also a parameter:
$
  "repeat" { \
    w_i^((j)) = w_i^((j)) - alpha partial / (partial w_i^((j))) J(w, b, x) \
    b^((j)) = b^((j)) - alpha partial / (partial b^((j))) J(w, b,x) \
    x_k^((i)) = x_k^((i)) - alpha partial / ( partial x_k^((i)) ) J(w, b,x) \
  }
$

=== Binary levels
*Definition:* This is collaborative filtering, but for binary labels, e.g click or not click, like or dislike.

*How it works:*
It is collaborative filtering, but instead of linear regression it becomes logistic regression, so you use:

$
  f_( (w, b, x) )(x) = g(w dot x + b)
$

$
  g(z) = 1 / ( 1 + e^(-z) )
$

To calculate loss function of a single example:

$
  L(f_( (w, b, x) )(x), y^((i,j)) )= \ - y^((i,j)) log (f_( (w, b, x) )(x)) - ( 1 - y^((i, j)) ) log (1 - f_( (w, b, x) )(x))
$

And for all examples:
$
  J (w, b, x) = sum_( (i, j):r(i, j) = 1 ) L ( f_(w, b, x) (x), y^((i, j)) )
$

=== Mean normalization

*Definition:* It is a preprocessing step where user's rating are adjusted.

*Problem it solves:*
- This center ratings around zero and removes individual rating vias.

*How it works:*

+ Create average array $mu$.
+ Subtract array $mu$ to initial items $y$, now this will be the new $y$
+ Use the normalized rating for prediction.
+ We need to sum $mu$ on logistic regression.
$
  w^((j)) dot x^((i)) + b^((j)) + mu_i
$

=== Finding similar items
We can use collaborative filtering with to find related items:

*How it works:*
To find other items similar to $i$ with features $x^((i))$ find item $k$ with $x^((k))$ that similar to $x^((i))$.

So we can use mean squared error to compare the features and get the one with the smallest error:

$
  sum_(l = 1)^n = ( x_l^((k)) - x_l^((i)) )^2
$

=== The Cold start problem
- How do we rank new items that few users have rated?
- How de we show something reasonable to users that do not have rated much movies?

In this case you use side information about items or users:
- Item: genre, movie start, studio.
User: Demographics (age, gender, locations), expressed preferences (like asking users in a form)


#pagebreak()

== Content based filtering
*What it is?*
A recommendation technique based on the relationship between user preferences and item features.

*Problem it solves:*
- The cold start problem: It does not depend on how many users have interacted with the item.
- Transparency: It is easy to explain why an item was suggested.

*How it works?*
+ We compute two vectors $v_u^((j))$ (user network) and $v_m^((i))$ (item vector) using neural networks. (It is important that both vectors are from the same size)
+ We can use dot product between both vectors to know similarity.
  - We can also apply sigmoid between those vectors to predict probability that $y^(i, j)$ is 1
+ Then we use cost function, we can make a set of trainings and the result is  the one with the least cost function.
$
  J = sum_( (i, j):r(i ,j) = 1 ) ( v_u^((j)) dot v_m^((i)) - y^(i, j)^2 ) + N N "regularization term"
$

Where:
- $v_u^((j))$: user vector of user $j$.
- $v_m^((i))$: item vector of item $i$.
- $y^((i, j))$: the real value model is trying to predict (depending on the system it may be: a real rating, a like, watching time, etc.)

=== Finding similar items
After training the model, we can also use it to find similar items:

*How it works?*
To find and item $k$ similar to $i$ we need to verify that distance between
$
  || v^((k))_m - v_m^((i)) ||^2
$

Where:
- $v^((k))_m$: item vector of item $k$.
- $v_m^((i))$: item vector of item $i$
Remember:
- $v_m$: item vector

You can pre-compute this ahead of time.

=== Managing large datasets
In order to manage large datasets we use two steps: Retrieval & Ranking.

==== Retrieval
*How it works?*
+ Generate large list of plausible item candidates. Example:
  - For each of the last 10 movies watched by the user find 10 mist similar movies $ || v^((k))_m - v_m^((i)) ||^2 $
  - For most viewed 03 genres, find the top 10 movies.
  - Top 20 movies in the country.
+ Combine items into a list and remove duplicated and already watched/purchased items.

*Recommendations:*
+ Retrieving more items results in better performance, but slower recommendations.
+ To analyze/optimize the trade-off, run offline experiments to see if retrieving additional items results in more relevant recommendations (i.e, $p(y^((i, j))) = 1$ of items displayed to user are higher)

==== Ranking
*How it works?*
+ Take list retrieved and rank using the previous learned model.
+ Display ranked items to the user.

== Ethical use - recommendations
+ Filter out problematic content such as hate speech, fraud.
+ Be transparent with users.

== Principal Component Analysis (PSA)
*What it is?*
- An unsupervised learning technique used to reduce the number of features of a dataset that still contains most of the original information.

*Problem it solves:*
- When a model is too large, it becomes computationally expensive, prone to overfit and hard to visualize.
- Reduces redundancy of the dataset.

*How it works?*
+ Normalize data to have zero mean.
+ Choose a new $z$ axis (this will be our new principal component) and project examples onto axis $z$.
  - We should choose one $z$ axis and do the projection
  #figure(caption: "Projecting points to axis z")[
    #image("images/project-z-axis.png")
  ]
  - There are many options for $z$ axis, we should choose the one that generates the max variance.
  #figure(caption: "Z axis with high variance")[
    #image("images/z-axis-with-max-variance.png")
  ]
  - We get the length of the $1$ vector, we get the coordinates of the point and do the dot product.
  $
    "point coordinates" dot "length 1 vector" = "value of projection"
  $
  #figure(caption: "Calculate projection")[
    #image("images/calculate-projection.png")
  ]

More techniques:
+ In order to get more principal components:
  - A 2nd axis is 90° of axis 1.
  - A 3rd axis is 90° of axis 1 and 2.
  #figure(caption: "More principal components z_2 and z_3")[
    #image("images/more-principal-components.png")
  ]
+ For approximate reconstruction:
  $
    "reconstruction" = "distance of projection" times "length 1 vector"
  $
