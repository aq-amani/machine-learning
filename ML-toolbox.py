import numpy as np
import matplotlib.pyplot as plt

def read_data(filename, feature_count):  
    """
    Reads comma separated data file and returns features and targets based on feature_count
    """
    dataset = np.loadtxt(filename, delimiter=',')
    X_train = dataset[:,0:feature_count]
    Y_train = dataset[:,-1]
    return X_train, Y_train
    
def plot_binary_classification_data(X_train, Y_train, good_label='pass', bad_label='fail', boundary=None, scatter_boundary=False):
    """
    Plots binary classification training data that has 2 features
    """
    X1_train = X_train[:,0] # First feature data
    X2_train = X_train[:,1] # Second feature data

    plt.scatter(X1_train[Y_train==1], X2_train[Y_train==1], marker = 'o', edgecolors='lime', facecolors='none', label=good_label)
    plt.scatter(X1_train[Y_train==0], X2_train[Y_train==0], marker = 'x', c='r', label=bad_label)
    if boundary is not None:
        if scatter_boundary:
            plt.scatter(boundary[:,0], boundary[:,1], label='Prediction boundary', s = 5)
        else:
            plt.plot(boundary[:,0], boundary[:,1], label='Prediction boundary')

    plt.legend(loc='upper right')
    plt.style.use('dark_background')
    plt.title("X1-X2 scatter plot")
    plt.xlabel('X1_train')
    plt.ylabel('X2_train')
    plt.show()
    
def sigmoid(Z):
    """
    g(Z) = 1/(1 + e^-Z)
    """
    return 1/(1 + np.exp(-1 *Z))

def linear(Z):
    """
    g(Z) = Z
    """
    return Z

def model_accuracy(Y, Y_hat, boundary_threshold=0.5):
    """
    Returns accuracy of a binary classification model
    """
    # Threshold Y_hat over 0.5
    P = np.where(Y_hat < boundary_threshold, 0, 1)
    #Ratio of correct guesses
    accuracy = np.mean(P==Y) * 100
    return accuracy
    
def prediction_function(X, w, b, activation=linear):
    """
    w can be an array of weights depending on the number of features in X
    Returns Y_hat (the predicted values) for X based on b and w parameters 
    y_hat = g(z) = 1/(1 + e^-z) for Logestic Regression (Z= w*X + b)
    """
    
    
    if X.ndim > 1:
        # More than 1 feature
        # Z = w1*X1 + w2*X2+ w3*X3 ... + b
        # axis =1 to sum rows
        Z = np.sum(w * X, axis=1) + b
    else:
        # One feature case
        Z = w * X + b
    
    # Call the passed activation function to get the predicted value(linear by default)
    Y_hat = activation(Z)
    return Y_hat

def cost_function(Y_hat, Y):
    """
    Returns the cost J based on the Y_hat(prediction) and Y (ground truth) data
    """
    J = np.mean(-Y * np.log(Y_hat) - (1-Y) * np.log(1-Y_hat))
    return J

def regularized_cost_function(Y_hat, Y, w, lambda_reg=0):
    """
    Returns the regularized cost J
    """
    reg_term = 0.5 * lambda_reg * 1/Y.shape[0] * np.sum(w ** 2)
    J = cost_function(Y_hat, Y) + reg_term
    return J

def gradient_descent(X, Y, w, b, alpha, iteration_count, lambda_reg=0, log_verbosity=1000):
    """
    Implements the Gradient Descent algorithm.
    Returns a log of the GD iterations with the optimum values [iteration num, cost, w, b] in the last entry
    Need to update each w1, w2, w3 ..etc for the case of multiple features
    """
    GD_log = np.empty((0,4), dtype=float)
    for i in range(iteration_count+1):
        Y_hat = prediction_function(X, w, b, activation=sigmoid)
        cost = regularized_cost_function(Y_hat, Y, w, lambda_reg)
        GD_log =  np.vstack([GD_log, np.array([i, cost, w, b], dtype=object)])
        # Print a log message every 1000 iteration
        if(i%log_verbosity ==0):
            print(f'Iteration #{i}: cost={cost:.2f}\tw={w}\tb={b:.2f}, ')
        Y_error = (Y_hat - Y)
        #Transoform Y_error a bit to make multiplication with multi-dimensional X possible
        Y_error = np.transpose([Y_error] * X.shape[1])
        
        dJ_dw = np.mean(Y_error * X, axis=0)
        dJ_dw += (lambda_reg * w) / X.shape[0]
        dJ_db = np.mean((Y_hat - Y))
        
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db
    optimal_w = GD_log[-1,2]
    optimal_b = GD_log[-1,3]
    print(f'Optimal parameters: Iteration {GD_log[-1,0]}, cost: {GD_log[-1,1]}, optimal w: {optimal_w}, optimal b: {optimal_b}')
    return optimal_w, optimal_b

def polynomial_feature_mapper(X, degree):
    """
    maps two features X1, X2 to multiple polynomial features of
    a maximum degree equalt to the passed degree
    X1,X2 --> X1, X2, X1**2, X1*X2, X2**2, X1**3, X1**2,X2 ....
    """
    powers = np.empty((0,2))
    count = 0
    for i in range(1,degree+1):
        for j in range(i+1):
            powers = np.vstack((powers,[i-j,j]))

    extended_X = np.empty((X.shape[0],powers.shape[0]))
    for i in range(powers.shape[0]):
        Y = np.prod(X ** powers[i,:], axis=1)
        extended_X[:,i] = Y

    return extended_X
