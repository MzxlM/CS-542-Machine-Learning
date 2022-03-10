from typing import Tuple
import numpy as np


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Xinlong Zhang", "BU_ID": "U00000000", "BU_EMAIL": "xinlongz@bu.edu"}

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        return X, y

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################

        X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        self.theta = np.random.rand(X_train.shape[1], 1)

        # result = self.run_gd(self.loss, self.loss_gradient, X_train, y_train, self.theta)
        result = self.run_ridge_regression(self.ridge_regression, X_train, y_train, self.theta)
        # theta_est, loss_values, theta_values = result
        theta_est, theta_values = result
        self.theta = theta_est

    def predict(self, X_val: np.array) -> np.array:
        """
            Predict with model and given feature
            """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        # print("X_val.shape", X_val.shape)
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        return np.dot(X_val, self.theta)

    def predicts(self, X, theta):
        # Xa = self.add_column(X)
        # print('X.shape: ', X.shape)
        # print('Xa.shape: ', Xa.shape)
        # print('theta.shape: ', theta.shape)
        return np.dot(X, theta)

    def add_column(self, X):
        return np.insert(X, 0, 1, axis=1)

    def loss(self, X, y, theta):
        return ((self.predicts(X, theta) - y) ** 2).mean() / 2

    def loss_gradient(self, X, y, theta):
        # X_prime = self.add_column(X)
        loss_grad = ((self.predicts(X, theta) - y) * X).mean(axis=0)[:, np.newaxis]
        return loss_grad

    def run_gd(self, loss, loss_gradient, X, y, theta_init, lr=0.07, n_iter=1000):
        theta_current = theta_init.copy()
        loss_values = []
        theta_values = []
        for i in range(n_iter):
            loss_value = loss(X, y, theta_current)
            theta_current = theta_current - lr * loss_gradient(X, y, theta_current)
            loss_values.append(loss_value)
            theta_values.append(theta_current)
        return theta_current, loss_values, theta_values

    def run_ridge_regression(self, ridge_regression, X, y, theta_init, lambda_=5.7, n_iter=1):
        theta_current = theta_init.copy()
        # loss_values = []
        theta_values = []
        for i in range(n_iter):
            # loss_value = loss(X, y, theta_current)
            theta_current = ridge_regression(X, y, lambda_)
            # loss_values.append(loss_value)
            theta_values.append(theta_current)
        return theta_current, theta_values

    # version 2
    def ridge_regression(self, X, y, lambda_):
        XTX = np.dot(X.T, X)

        print('shape ', XTX.shape)
        I = np.matrix(np.eye(XTX.shape[1]))
        # I = np.ndarray(np.eye(m))

        return (XTX + lambda_ * I).I * X.T * y
