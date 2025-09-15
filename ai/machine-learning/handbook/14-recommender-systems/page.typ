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
