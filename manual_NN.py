import numpy as np

def initialize_parameters_deep(layer_dims, layer_size_relative_scaling=False):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    layer_size_relative_scaling -- whether to scale W random values based on previous layer size or not
    (will scale by *0.01 by default)

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        if layer_size_relative_scaling:
            parameters['W' + str(l)] /= np.sqrt(layer_dims[l-1])
        else:
            parameters['W' + str(l)] *= 0.01

    return parameters

# Activation functions
def sigmoid(Z):
    """
    sigmoid activation

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """

    A = 1/(1+np.exp(-Z))
    assert(A.shape == Z.shape)

    return A

def relu(Z):
    """
    RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)

    return A

def tanh(Z):
    """
    tanh activation function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.tanh(Z)
    assert(A.shape == Z.shape)

    return A

# Linear
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.matmul(W, A) + b

    cache = (A, W, b)

    return Z, cache

# Activation
def activation_forward(Z, activation):
    """
    Implement the activation function g

    Arguments:
    Z -- the output of the linear forward function
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid", "relu" or "tanh"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing the "activation_cache"
    """
    cache = Z
    if activation == "sigmoid":
        A = sigmoid(Z)

    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)

    return A, cache

# Controller for forward propagation
def forward_propagator(X, parameters, h_layer_activations="relu"):
    """
    Implement forward propagation for the [LINEAR->g(z)]*(L-1)->LINEAR->SIGMOID computation
    hidden layer activations are relu by default.

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep(). Can also be used to get the layer count.
    h_layer_activations -- activation type for hidden layers. relu by default

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # [LINEAR -> g(z)]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        # Linear function
        Z, linear_cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        # Activation function
        A, activation_cache = activation_forward(Z, h_layer_activations)
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    # LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # Linear function
    Z, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    # Activation function
    AL, activation_cache = activation_forward(Z, "sigmoid")
    cache = (linear_cache, activation_cache)
    caches.append(cache)

    return AL, caches

# Logistic Regression cost function (cuz last layer is a sigmoid one)
def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    cost = np.squeeze(cost)


    return cost

# Derivative of the Logistic Regression cost function
def cost_function_derivative(Y, AL):
    """
    dAL (dJ/dAL : last layer) is the derivative of the logistic regression
    cost function with respect to AL (= y_hat)
    """
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    return dAL

# g'(Z) depending on activation type
def activation_function_derivative(Z, activation_type):
    """
    Returns derivative of the activation function of a particular layer l(?)
    gl`(Zl)
    """
    if activation_type == "relu":
        # g(Z) = max(0, Z) --> derivative is 1 when Z >0 and is 0 otherwise
        g_prime = np.where(Z > 0, 1, 0)

    elif activation_type == "sigmoid":
        # derivative of sigmoid (s) is  s * (1-s)
        s = sigmoid(Z)
        g_prime = s * (1-s)
    elif activation_type == "tanh":
        # Derivative of tanh(x) is (1 - tanh^2(x))
        t = tanh(Z)
        g_prime = 1 - np.power(t, 2)

    return g_prime

# dW, db, dA_prev
def get_gradients(dA_l, cache, activation):

    linear_cache, activation_cache = cache
    Z_l = activation_cache
    m = Z_l.shape[1]
    A_prev, W_l, b_l = linear_cache
    g_prime_l = activation_function_derivative(Z_l, activation)
    dZ_l = dA_l * g_prime_l

    dA_prev_l = np.matmul(W_l.T, dZ_l)
    dW_l = 1/m * np.matmul(dZ_l, A_prev.T)
    db_l = 1/m * np.sum(dZ_l, axis=1, keepdims=True)

    return dA_prev_l, dW_l, db_l

# Controller for backwards propagation
def backward_propagator(A_L, Y, caches, h_layer_activations="relu"):
    """
    Backward propagation for a [LINEAR->g(z)] * (L-1) -> [LINEAR -> SIGMOID] NN

    Arguments:
    A_L -- probability vector, output of the forward propagation (also equals y_hat)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with g(z) (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    h_layer_activations --  activation type for hidden layers. relu by default

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = A_L.shape[1]
    Y = Y.reshape(A_L.shape) # after this line, Y is the same shape as AL

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    dA_L = cost_function_derivative(Y, A_L) # derivative of cost with respect to AL
    cache_L = caches[-1] # cache for last layer
    dA_prev_L, dW_L, db_L = get_gradients(dA_L, cache_L, "sigmoid")

    grads["dA" + str(L-1)] = dA_prev_L
    grads["dW" + str(L)] = dW_L
    grads["db" + str(L)] = db_L

    # From l=L-1 to 1
    for l in reversed(range(1, L)):
        # lth layer: (g(z) -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]

        cache_l = caches[l-1] # list indices are 0~L-1, while layer naming is 1~L (for layer l, index is l-1)
        dA_l = grads["dA" + str(l)]

        dA_prev_l, dW_l, db_l = get_gradients(dA_l, cache_l, h_layer_activations)

        grads["dA" + str(l - 1)] = dA_prev_l
        grads["dW" + str(l)] = dW_l
        grads["db" + str(l)] = db_l

    return grads

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def train_L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, h_layer_activations="relu", layer_size_relative_scaling=True):
    """
    Training full cycle for an L-layer neural network: [LINEAR->g(z)]*(L-1)->LINEAR->SIGMOID.
    Initialize then iterate over forward_prop -> cost -> backward_prop -> update_parameters (gradient descent)

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    h_layer_activations --  activation type for hidden layers. relu by default
    layer_size_relative_scaling -- whether to scale W random values based on previous layer size or not
    (will scale by *0.01 by default)

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims, layer_size_relative_scaling)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> g(z)]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = forward_propagator(X, parameters, h_layer_activations)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = backward_propagator(AL, Y, caches, h_layer_activations)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def predict(X, y, parameters, h_layer_activations="relu"):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    y_predictions -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    y_predictions = np.zeros((1,m))

    # Forward propagation
    # y_hat is probabilities
    y_hat, caches = forward_propagator(X, parameters, h_layer_activations)

    # convert y_hat to 0/1 predictions
    y_predictions = np.where(y_hat > 0.5, 1, 0)

    print("Accuracy: "  + str(np.sum((y_predictions == y)/m)))

    return y_predictions
