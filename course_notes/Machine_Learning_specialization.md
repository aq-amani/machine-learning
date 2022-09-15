My notes of Andrew Ng's Machine Learning specialization courses

## AI
- A.N.I.: Artificial Narrow Intellegence: Specific application like self-driving cars, factory defect detection, AI in farming..etc
- A.G.I.: Can do anything that humans can do. Not limited to specific domains.

**The One Learning Algorithm Hypothesis**: One algo to learn anything. Inspired by the fact that the same brain tissue can learn to "see" or to "hear" ..etc depending on what data is fed to it

# Machine Learning Types
High level map of ML types
## 👨‍🏫Supervised Learning
We have both data(features) and data labels(target)
#### Problem/Algorithm types
- Regression
  - Linear/Polynomial: continuous output | estimate curve that best fits data and use it to predict output of new data.
  - Decision trees
  - Random Forest
- Classification: Discrete output
  - Logestic Regression
## 🧠Un-Supervised Learning
Find patterns and structure in data(features) without labels available before hand
#### Problem/Algorithm types
- Clustering
  - K-means 
- Anomaly Detection
- Dimensionality reduction(?)
## 👾Reinforcement Learning (RL)
- `Agent` in an `Environment` with a `state`, `actions` and `Rewards` (both +ve and -ve)
- Agent works towards maximizing reward

# Details
Detailed description of the Algorithms and other concepts
## Generic form of Cost Functions
The cost function is the average Loss over all the training data set.
```
J = 1/m * Σ L(y_hat, y)
```
where m is number of data examples/instances and L is the `Loss Function` (different function  depending on the problem type)
## Linear Regression (single feature: univariate)
For linear or roughly linear data
1) Let the prediction/model function be 
   ```
   y_hat = f(x) = w*x + b
   ```
   where x is data/features, w is `weight` and b is `bias` (w, b: Parameters of model)

2) Let the `Cost function` J be the `Squared Error Cost function` (or `Mean Squared Error (MSE)`) defined as
   ```
   J = 1/m * Σ 1/2*(y_hat - y)^2
               ~~~~~~~~~~~~~~~~~~ ----> Loss function for squared error cost
   ```
   where m is number of training examples

   ℹNote : Cost function is a 3D plot of w and b, and the squared error cost function gives a bowl shaped curve.
   
   This is good because it wont have many local minimums one can "fall into". Just one clear global minimum.
3) Find `w` and `b` that minimizes the cost function using an `optimization algorithm` called `Gradient Descent (GD)`

#### Gradient Descent Algorithm
1) Choose the initial w and b roughly
2) Keep updating w and b values simultaneously as follows, until convergence
```
w_new = w - α * ∂/∂w J(w,b)
b_new = b - α * ∂/∂b J(w,b)
```
where `α` is the `Learning Rate` : corresponds to step size downhill on each iteration

the partial derivate corresponds to the `direction` to choose to go downhill the curve, and resolves to the following
```
∂/∂w J(w,b) = 1/m Σ (w*x + b - y) * x
∂/∂b J(w,b) = 1/m Σ (w*x + b - y)
```
#### Multiple Linear Regression (multiple features per data instance/example)
- Weight vector W : w1, w2, w3 .. etc for each single feature
- Feature vector X : x1, x2, x3 ..etc data value for each feature in each data instance/example
- Prediction function will be: 
   ```
   F(X) = w1*x1 + w2*x2 + w3*x3 +... + b
   ```
which can be implemented as `np.dot(W, X) + b` with numpy, where the dot product gives a scalar output
- GD will update all weights(w1, w2, w3...) and b
#### Feature Scaling
- Done when the scale of different features is so different( ex.: 2 < x1 < 5, 0 < x2 < 100M)
- Scales all features to have close ranges, and to enhance Gradient Descent speed
- Methods
  - Divide each feature by the maximum value of that feature
  - **Mean Normalization** : 
      ```
      x_scaled = (x - μ) / (x_max - x_min)
      ```
      where μ is the mean (avg) of feature x values
  - **Z-score normalization** : 
      ```
      x_scaled = (x - μ) /σ
      ```
      where σ is the standard deviation of feature x values
#### Convergence test
- Plot curve of GD iteration number and the cost (J) : This is called the (a?) `Learning Curve`
- See when the curve flattens out
- Or define a small value `ε` where if a new iteration decreases J by less than it, then convergence is declared
#### Choosing the Learning Rate α
- Plot the learning curve with a number of different α values for a couple of GD iterations
- See what decreases J the fastest witout overshooting or oscillating issues
## Polynomial Regression
- When needing a curve rather than a line to better fit/model the data
- Powers/ roots of features (ex.: `f = w1*X + w2*X^2 + w3*X^3 + b` , `f = w1*X + w2*√X + b`)

## Binary Classification
Ex.: Cat or not? Diagnosis postive or negative?

Mainly uses the `Logistic Regression` algorithm based on the `Sigmoid function` (AKA Logistic function)
#### Logistic Regression
Gives a probability of the input being of a certain class
```
y_hat = g(z) = 1/(1 + e^-z)
```
where z is the model funtion `w*x + b`, and `0 <g(z)< 1`

TODO: Add pic of sigmoid function
#### Cost function for Logistic Regression
With logistic regression, using a Squared Error cost function will result in multiple local minimums and a non-convex function.

For logistic regression, use this cost function instead
```
J = 1/m * Σ L(y_hat, y)
L(y_hat, y) = {-log(y_hat) , if y=1
              {-log(1-y_hat), if y=0
```
This cost function gives a convex curve and is possible to reach global minimum using Gradient Descent

Loss function can be resolved as follows to make it easier for computation
```
L(y_hat, y) = -y * log(y_hat) - (1-y) * log(1-y_hat)
```
where log is the natural log (ln or log to base e). 

`log` in both numpy and math libraries mean natural log by default if no base is specified.

TODO: plot functions?

#### Gradient Descent for Logistic Regression
Same definition and algorithm as that of Linear Regression but the Cost function is replaced by the one for Logistic Regression
```
w_new = w - α * ∂/∂w J(w,b)
b_new = b - α * ∂/∂b J(w,b)
```
where
```
J = 1/m * Σ L(y_hat, y)
y_hat = g(z) = 1/(1 + e^-z)
z = w*x + b
```
#### Model Fitting issues
**Overfitting (High Variance)**: Model fits training data overly that it fails to generalize and give predictions to new unseen data (usually higher degree polynomial of feature with many features and not enough data)

**Underfitting (High Bias)**: Model doesn't fit training data well (ex.: using a line when a higher degree curve is needed to fit data)

Dealing with overfitting (high variance):
- More training data
- Feature selection/reduction: Reduce the features to the most relevant ones
- `Regularization`: Reduce the size of the parameters(w and b) by using a `Regularization parameter (λ)` while keeping all features. Done by adding the following regularization term to the cost function:

$$\\frac{\\lambda}{2m}  \\sum_{j=0}^{n-1} w_j^2$$

Dealing with underfitting (high bias):
- More features per data example
- Add polynomial features
- Smaller `Regularization term (λ)`
- Larger Neural Network (more layers and more neurons)

## Neural Networks (NN)
Multiple regression/classification units connected over several layers

NN architecture components:
- Hidden Layer count (layer count doesn't include input layer)
- Neuron count per layer
- Each neuron has `w`, `b` and `a` (`activation`/result) values, with an `activation function` type
  ```
  a = g(z) = 1/(1 + e^-z)
  z = w*x + b
  ```
  The `a` vector of a layer is that layer's output and is the next layer's input
- Neuron parameters (w, b, a) are indexed by the layer number, and by the neuron number within that layer
  ```
  For layer l and neuron n
  a[l,n] = g(w[l,n] * A[l-1] + b[l,n])
  ```
  where `A[l-1]` is a vector of all a1,a2.. of the previous layer
  
    - If `A[l-1]` is of size `mx1`, and there were `n` neurons in layer `l`, then `W` is of size `nxm`.
    - `W*A + b` is done using `np.matmul(W, A) + b` and output will be of size `nx1`.
    - Can also be done as `np.sum(A.T * W, axis=1) + b`, where `*`(or `np.multiply()`) is the element-wise multiplication.
  
- **Forward Propagation(inference/prediction)**: The output layer will give a probability of what we are checking against. Can use a threshold(ex.: 0.5) to decide.
- **Back Propagation**: The process of updating w, b and computing derivatives while doing Gradient Descent.

**Example: Object recognition in images**
- Feed image pixels as a vector to the NN input
- Usually first layer looks for edges
- Second for zoomed-out more complex shapes (eyes, nose..etc)
- Next layer for more complex objects (face, car ..etc)

And it figures out (learns) these features on its own

#### Layer types
- Dense layer: Every neuron takes all outputs of previous layer
- Convolutional layer: Every neuron looks at only part of the previous layer output (CNN: Convolutional Neural Network .. common with images) 
#### Activation Functions
Most common:
- Linear Activation function: `g(z) = z` Does nothing just passes things through
- ReLU: Rectified Linear Unit `g(z) = z if z >=0 else 0`/ `g(z)=max(0,z)`
- Sigmoid : `g(z) = 1 / (1 + e^-z)`

How to choose function for `Output layer`:
- Binary Classification problem? Sigmoid
- Regression problem where -ve values are ok ? Linear
- Regression where y can only be positive? ReLU

How to choose function for `Hidden layers`:
- ReLU is the most common choice regardless of problem type (faster and easier for GD)

#### Adaptive Moment Estimation Algorithm (ADAM)
- An algorithm to speed up Gradient Descent by adaptively increasing/decreasing the learning rate α
- Has multiple α , one for each `wi` and `b` (TODO: details?)

## Multi-class Classification
For problems where the output can have more than one possible outcome (ex.: cat? dog? human?)

Uses the `Softmax Regression Algorithm`, a generalization of Logistic Regression.

#### Softmax Regression
Define a `z` for each possible output class. For k possible output classes and for the ith class:
```
z_i = W_i * X + b_i
a_i = e^z_i / (Σ e^z on all k)
```
Example: For the 3rd class in a 4 classes problem:
```
a_3 = e^z_3/ (e^z_1 + e^z_2 + e^z_3 + e^z_4)
```
NN output layer will have k neurons, one for each class.

Loss Function:
```
L = { -log(a_1) , if y=1
    { -log(a_2) , if y=2
    { -log(a_3) , if y=3
         ....
    { -log(a_k) , if y=k
```
## Multi-label Classification
- Don't confuse with multi-class.
- Example problem: For a given single image, does it have cars? busses? pedestrian? (these are the labels)
- Output is a vector of the probabilities of existence of the different labels
- Can be done with
  - multiple NNs, one for each label
  - OR better approach: One NN, multiple outputs with one for each label

## Model Evaluation and Diagnostics
`Generlization errors` : J_test obtained from test data set and `Training error`: J_training
#### Evaluating a single model
- Split data into a training set and a test set
- Evaluate as follows:
  - Regression problem: Compare J_training and J_test (training data cost function and test data cost function)
  - Classification problem: See the fraction of training set and test set that the model mis-classified
#### Evaluating multiple models or NN architectures to choose from
- Split data into 3 sets: a training set, a `cross validation set (CV set|dev set|validation set)` and a test set
- Choose model with lowest CV cost or CV error (J_cv)
- test set is only to be used as the last step after the model choice and optimization is done
#### Bias and Variance for evaluation/diagnostics
When dealing with fitting issues, either tackle the Model (model centric approach) or the Data (data-centric approach), or both.
- J_training | J_cv
  - High | High : Underfitting (high bias)
  - Low | High : Over fitting (high variance)
  - Low | Low : Good
- Can plot J_train and J_cv against different polynomial degrees or against the trainig data size and see how they affect it
- More training data only helps with overfitting but not with underfitting problems
- A larger NN (more layers and neurons per layer) helps with high bias (underfitting) issues
#### Some techniques
- `Data Augmentation`: Create additional training data by modyfying/distoring existing data (flip/tilt/zoom/noise..for images, and audio in different noise types crowd/traffic/bad cellphone connection.. for audio)
- `Data Synthesis`: Artificial data inputs for new training data (screenshotting computer fonts)
- `Transfer Learning`: Reusing NN of another task as a starting point for the current task when not much data is available (input type must be the same)
  - How? Either Replace the output layer only OR retrain the NN with the previous task parameters as a starting point
  - Ex.: Using an NN that can classify 1000 categories of things, to recognise hand written digits while optimizing for output layer only
Optical Character Recognition (OCR): Images of text to text data
#### Skewed data sets, Precision, Recall and F1 Score
Data sets that are not balanced for all the different output categories

Ex.: Rare disease classification problem. Most of the data will be negative diagnosis examples.

`Precision`, `Recall` and `F1 score` are better error metrics for such data
```
Precision (P) = True Positives/Predicted Positives = True Pos/(True Pos + False Pos)
Recall (R) = True Positives/ Actual Positives = True Pos/(True Pos + False Neg)
F1 score = 1/(0.5* (1/P + 1/R) ) = 2PR/(P+R)
```
- Must aim to maximize both P and R.
- Higher detection threshold increases Precision but lowers Recall. Opposite with Lower detection threshold.
- **F1 Score** is used to decide the best combination of P and R when evaluating multiple algorithms, models, thresholds or NN settings. (Pick config that gives highest F1 score)

## TensorFlow / Keras notes
- `BinaryCrossentropy()` --> Logistic Regression Loss function for binary classification
- `SparseCategorialCrossEntropy()` --> Softmax Regression Loss function for multi-class classification
- `MeanSquaredError()` --> Regression Loss function
- epochs : How many times to go through training data. Can be thought of as the number of iterations in Gradient Descent.
- Utilize `np.dot()` and `np.matmul()`
- Python Precision related: To get more accurate results with softmax and logistic regression (TODO: only those?):
  - set `from_logits = True` in the loss function
  - set output layer activation function to `linear`. With this `model(x)` wont give us probabilities anymore, so to get them do as in next step
  - ```
    model.fit(....)
    logit = model(x)
    f_x = tf.nn.sigmoid(logit)
    OR
    f_x = tf.nn.softmax(logit)
    ```
 - Using ADAM to speed up Gradient Descent
 ```
 model.compile(optimizer=tf.keras.optimizers.Adam(..,learning_rate=XX, ..))
 ```
 - Regularization: can be specified as a parameter to the layer:
 ```
 Dense(units=xx, activation="relu", kernel_regularizer=L2(lambda_value))
 ```
 ## Decision Trees
 Uses `Entropy` as a measure of impurity. 
 
 The more or the less of one class in the sample, the lower the entropy. The closer the split is to 50%, the higher the entropy.
 
 Example: For a cat/dog classification problem, Entropy is a function of the fraction of cats at a certain node split.
 
 #### Basic algorithm
Keep splitting while choosing features that give the highest `Information Gain` until a stop condition is reached. Examples:
- A node is 100% of one class
- A predefined max tree depth has been reached
- Further splits are not giving much more information gain (based on a set threshold)
- Number of examples in a node became less than a certain threshold

 #### The Entropy function H(p)
 ```
 H(p1) = -p1 * log2(p1) - p0 * log2(p0)
 p1 = number of class1 objects / total number of objects
 ```
 where p1 is the fraction of one class (ex.:cats) and p0 is the fraction of the other(dogs | not cat)
 TODO: Plot H(p)
 
 #### Information Gain
 Less Entropy gives more `Information Gain`
 
 For N total samples in the parent node, if the split gave n1 and n2 samples respectively in each node:
 ```
 Information gain = Entropy of parent node - weighted average of the new Entropies of each node
 Information Gain = H(N_i/N) - ( n1/N * H(n1_i / n1) + n2/N * H(n2_i/ n2))
 ```
 Where n1_i, n2_i is the number of class i samples in each new node, and N_i is number of i class samples in the original node
 
 Choose features to split over based on the one that gives the highest information gain

#### Some techniques
- `One hot encoding`: When a feature has more than two possible values, replace it with multiple features each taking either 0 or 1
- `Tree Ensemble` : Train multiple different trees to get higher accuracy since trees are sensitive to small changes in training data.
- `Sampling with replacement`: Shuffle training data while allowing duplicates and then train the tree on the resulting new data sets
- `Random Forest` : Choose random subsets of features to split upon. 
   TODO:more elaboration
- `Boosted Trees` : Sampling with replacement but with more probability of picking data that was mis-classified in previous trees.
  - `Extreme Gradient Boosting (XGBoost)` is an implementation of boosted trees

#### Descision Trees VS NNs
Trees work well on structured data (tabular data) but not with unstructured data (images, audio, text)

NNs are good with both data types

Trees are faster than NNs

## Clustering

#### The K-means algorithm
K is the number of clusters we want to group the data into. 

Choice of K: based on a tradeoff between accuracy and the final application complexity or cost(ex. not helpful to have many t-shirt sizes)
1) Choose K random data points as the initial cluster `centroids`
2) Assign points to clusters based on the distance between the point and the centroids (assign to closest cluster)
3) Update centroids to the average of the points that got assigned to them. Repeat step 2 and 3.
    ```
    J = 1/m * Σ | x - μ |^2
    ```
    where μ is the centroid, `|a - b|^2` is the `Squared Euclidean distance` between two points a and b.

    This J is also called the `Distortion cost function`

4) Repeat for different initial centroids and choose the config with the lowest J.
   - This is because the initial centroid choice affects the final cluster assignment

## Anomaly Detection

1) Find `μ`(mean) and `σ^2`(variance) of `normal` data samples for each feature
   - Can plot a histogram of the data and see if it roughly follows a normal distribution
   - If not, need to `transform the data` by trying `log(x)`, `log(x+c)`, `x^1/2`, `x^1/3`. 
     - Must apply same transformation to all data including CV and test data. 
2) Costruct a Gaussian probability distribution graph based on that `μ`and `σ^2`
   - `μ` is where the graph is centered symmetrically, `σ^2` is how spread the graph is around the center.

   $P(x) = \\frac{1}{\\sqrt{2 \\pi \\sigma ^2}}\\exp^{ - \\frac{(x - \\mu)^2}{2 \\sigma ^2} }$  

3) Calculate P(x) for the data to check
4) If `P(x) < ε` then declare anomaly
   - `ε` is a small probability value of which to be chosen/tuned based on anomalous examples

ℹ Note: If the model is not distinguishing well between anomaly and normality due to p(x) being close in both cases, try to add a new feature that is more prominent in anomalous examples.

## Reinforcement Learning

`Agent` in an `Environment` with `states`, possible `actions` and `Rewards` (both +ve and -ve)

State: `s`, Action: `a`, next state: `s'`, Reward function: `R(s)`

`Terminal state`: State(s) at which one round of learning ends

#### Deep Q-Network (DQN) Algorithm
The goal is the learn the Q function
1) Define the `state vector` and the `rewards` for each state
2) Start with a random guess of Q(s,a)
3) Take many actions randomly and store the resulting data (s, a, R(s), s')
    - `Replay Buffer`: How many most recent N data tuples to store 
4) Create a training data set where `X=(s,a)`, `Y= Q(s,a) = R(s) + γ * MAX( Q(s', a') )` (Bellman equation)
    - For Q(s',a'), choose a random initial value.
5) Train a NN such that Q_new roughly equals Y
6) Set Q = Q_new

ℹ Note: The NN needs to be trained for each action separately if it has one output. So a better way is to have one NN with action count number of outputs and train it once.
#### Return Function
Function of the Reward with a `Discount Factor (γ)` (gamma) to make the algorithm "impatient"
  - Discount factor role: Minimize steps to reach rewards by reducing reward value in proportion to the number of steps
  - maximize rewards in least steps or time ticks
```
Return = R1 + γ * R2 + γ^2 * R3 + γ^3 * R4 .. until terminal state
```
where Rn is the `Reward` of state n. γ is usually chosen close to 1 (0.9, 0.99 ..etc)

#### Policy function
Policy function `Π(s) = a` , Maps a State to an Action: What action to take in a given state.

This formalism is referred to as a `Markov Decision Process (MDP)`: Future depends only on the current state.

#### The Q-function (State-Action value function)
The `State-Action value function` or the `Q function` is defined as
```
Q(s,a) = Return , when starting at 's', taking 'a' only once and behaving optimally after that
```
The best possible return of state s is the maximum return when trying different actions
```
Best Return = MAX(Q(s,a)) over the different actions
```
The Q function can be calculated using the `Bellman Equation`:
```
Q(s,a) = R(s) + γ * MAX( Q(s`, a`) )
```
#### ε Greedy Policy
`ε Greedy Policy` is a way to choose actions while having both `Exploitation` and `Exploration` by assigning probabilities to each.
- `Exploration`: Random action with a small probability of `ε` (ex.:0.05)
- `Exploitation`: Pick action that maximizes `Q(s,a)` with a bigger probability (ex.: 0.95)

ℹ It's possible to start with a high `ε` and decrease it with time to favor exploration early in the process.

#### Stochastic (random) environments
- Actions will fail with certain probabilities and the agent will end up in different states than intended
- In such environments, aim to maximize the `expected` returns (average of returns over many attempts? TODO: check)
