from typing import Tuple
import numpy as np


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "", "BU_ID": "UXXXXXXXX", "BU_EMAIL": "XXX@bu.edu"}

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

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        return np.dot(X_val, self.theta)
