# We will build a linear regression model from scratch to understand:
    # how the model calculates the parameters(weights and bias)
    # the different hyperparameters present(epochs , learning rate and batch size)
    # gradient descent
    # the equation of a straight line , how to calculate loss and the different types of loss

import numpy as np

class linearregressionscratch:
    # dunder method init that outamtically executes when we call the class
    def __init__(self , learning_rate = 0.01 , epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weight = 0.0 # we initialize our weight to 0
        self.bias = 0.0 # we initialize our bias to 0

    # create a function for fitting the model

    def fit(self , x , y):
        # calculate the length of our features
        n_length = len(x)

        # we create a loop for our gradient descent

        for _ in range(self.epochs):
            # calculate the predicted y value and the error
            y_pred = (self.weight * x) + self.bias
            error = y - y_pred

            # calculate the respective slopes per parameter
            # weight slope
            slope_w = (-2 / n_length) * np.sum(error * x)
            # bias slope
            slope_b = (-2 / n_length) * np.sum(error)

            # after calculating the weight and bias slope we now update the weight and bias

            self.weight = self.weight - (slope_w * self.lr)
            self.bias = self.bias - (slope_b * self.lr)


    # create a function for predicting

    def predict(self , x):
        return (self.weight * x) + self.bias

# we can now go ahead and test our model

X_features = np.array([1.0 , 2.0  , 3.0 , 4.0 , 5.0])
y_features = np.array([2.5 , 4.5 , 6.5 , 8.5 , 10.5])


# we now instantiate our model , train and predict

model = linearregressionscratch(learning_rate = 0.01 , epochs = 1000)
model.fit(X_features , y_features)

# We can get the model weights and bias

print(f"Weight: {model.weight:.2f}")
print(f"Bias: {model.bias:.2f}")

# if you were to plot the x features and y_features in a paper you would expect to have the m(weight)= 2.0 and bias(y-intercept) = 0.5

# when you print we should expect to see the same


# SECOND METHOD TO COME UP WITH OUR LINEAR REGRESSION MODEL

# We use a linear algebra to calculate the weight and bias without going throught the hustle of comping up with epochs , learning rates and iterations. 

import numpy as np

# 1. Setup sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.5, 4.5, 6.5, 8.5, 10.5])

# 2. we add a column of 1s to the features (this handles the bias math automatically)
X_with_bias = np.c_[np.ones((len(X), 1)), X]

# 3. we apply the Normal Equation formula in one single line of code
best_weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)

# 4. Extract answers
print(f"Instant Bias: {best_weights[0]:.2f}")    # Outputs: 0.50
print(f"Instant Weight: {best_weights[1]:.2f}")  # Outputs: 2.00
