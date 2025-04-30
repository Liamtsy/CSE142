import numpy as np

class Perceptron:
  def __init__(self, max_iter = 50, eta = 1.0, shuffle = True, random_state = None):
    self.max_iter = max_iter # The number of epochs
    self.eta = eta # Learning rate
    self.shuffle = shuffle # Determine if the data is shuffled every epoch or not
    self.rng = np.random.RandomState(random_state) # To output different results when we use this perceptron multiple times, This is just a object to generate random numbers

  def _shuffle(self, X, y):
    """
    X: Features Matrix (n_samples x n_features)
    y: Labels vector (n_samples)
    Returns: X, y rows are randomly shuffled in the same order
    """
    idx = self.rng.permutation(len(y)) # Generate a random permutation of the indices of the samples, random order
    return X[idx], y[idx] # Return the shuffled X and y
  
  def fit(self, X, y):
    """
    X: 2-dimensional numpy array (n_samples x n_features), 1 e-mail = 1 sample
    y: 1-dimensional numpy array (n_samples). 0 or 1
    Returns: self
    """
    # Convert the labels to -1 and 1
    y_bin = np.where(y == 0, -1, 1).astype(float) # if the e-mail is not spam, label = -1. if the e-mail is spam, label = 1.
    # Also, np.where() is vectorized operation. so it works like iterating through all the samples and applying the condition.
    n_features = X.shape[1] # Get the number of features, .shape returns a tuple of the shape of the array. (Rows, Columns)

    # Initialize the weight vector and bias
    self.w = np.zeros(n_features) 
    self.b = 0.0 

    # Training Loop
    for epoch in range(self.max_iter): # For each epoch
      # if self.shuffle is True, the data is shuffled every epoch
      if self.shuffle:
        X, y_bin = self._shuffle(X, y_bin) # Shuffle the data

      errors = 0 # Count the number of errors

      for xi, target in zip(X, y_bin): # zip() is used to pair the samples and labels together
        linear_val = np.dot(xi, self.w) + self.b # Dot product of the sample and the weight vector, plus the bias
        if target * linear_val <= 0: # If the predicted label is incorrect
          self.w += self.eta * target * xi # Update the weight vector
          self.b += self.eta * target # Update the bias
          errors += 1

      if errors == 0:
        break
        
    return self # Return the trained model to predict the labels of the test set
  
  def print_weights_bias(self):
    print(f"Weights: {self.w}")
    print(f"Bias: {self.b}")
  
  def predict(self, X):
    """
    X: test data (n_samples x n_features)
    """
    linear = np.dot(X, self.w) + self.b # Dot product of the sample and the trained weight vector, plus the bias
    return (linear >= 0).astype(int) # Return the predicted labels, 1 = spam, 0 = not spam