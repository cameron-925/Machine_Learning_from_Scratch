import numpy as np

class LogisticRegression():
    def __init__(self):
        self.weights = None
        self.costs = None

    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig

    def fit(self, X, y, learning_rate = 0.01, iterations = 100, stopping_threshold = 1e-4):
        # Create Bias Column
        col_ones = np.ones(shape=(X.shape[0],1))
        X = np.concatenate((col_ones, X), axis = 1)

        # Reshape y
        y = y.reshape(-1, 1)

        # Initialize Weights
        self.weights = np.zeros((X.shape[1], 1))

        # Num Cols
        m = X.shape[0]

        # Costs
        costs = []
        cost = 99999 # initialize cost

        # Precompute Transpose
        X_transpose = X.T

        # Gradient Descent
        for i in range(iterations):
            cost_prev = cost

            # Compute Hypothesis Function
            h = self.sigmoid(np.dot(X, self.weights))
            h = np.clip(h, 1e-15, 1 - 1e-15)

            # Compute Gradient
            grad = (1/m) * np.dot(X_transpose, h-y)

            # Update Weights
            self.weights -= learning_rate * grad

            # Compute Cost
            cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            costs.append(cost)
            # print("Iteration:", i)
            # print("Cost:", cost)

            # Stopping Criteria
            if abs(cost_prev-cost) <= stopping_threshold:
                self.costs = costs
                break

        self.costs = costs

    def predict(self, X):
        # Create Bias Column
        col_ones = np.ones(shape=(X.shape[0],1))
        X = np.concatenate((col_ones, X), axis = 1)

        p = self.sigmoid(np.dot(X, self.weights))
        p = p >= 0.5
        p = p.astype(int)
        return p


def normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    return (X-mean)/std

def accuracy_score(y_pred, y_true, r=2):
    y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)

    acc = (y_pred == y_true).sum() / len(y_true)
    acc = round(acc, r)
    return acc