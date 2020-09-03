"""
Initialize sample weight as 1/N

For m=1 to L
1) Fit the classifier and get the best fit(smallest training error)
2) Get the classifier's training error (e)
3) Compute weight of the classifier
weight = (1/2) * np.log((1-e)/e)
4) set new sample weight and normalize it to 1

Weighted majority vote to calculate result
"""
from typing import List

import pandas as pd
import numpy as np
from copy import deepcopy

class AdaboostES:
    def __init__(self, voting_clf: List, max_iter, early_stopping: bool):
        ...
        self.clfs = voting_clf
        self.iter = max_iter

    def _validity_check(self, X_train: pd.DataFrame, y_train:pd.DataFrame) -> List:
        """
        Check the validity of the labels
        to use the loss function of AdaBoost
        which is exp(- y * y_hat)
        s.t y_hat = prediction, y = true value
        """
        y_content = list(y_train[y_train.columns[0]])

        assert set(y_content) == {-1, 1}
        return X_train, y_train

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        For m=1 to L
        1) Fit the classifier and get the best fit(smallest training error)
        2) Get the classifier's training error (e)
        3) Compute weight of the classifier
        weight = (1/2) * np.log((1-e)/e)
        4) set new sample weight and normalize it to 1
        """
        self._validity_check(X_train, y_train)

        # Get optimal classifier for the iteration
        num_feat = len(X_train.columns)
        num_data = len(X_train)

        # Number of iterations are needed to proceed with data
        self.sample_weights = np.zeros(shape=(self.iter, num_data))
        self.classifier = np.zeros(shape=self.iter, dtype=object)
        self.clf_weights = np.zeros(shape=self.iter)
        self.errors = np.zeros(shape=self.iter)

        # Set initial weight(all equal)
        self.sample_weights[0] = np.ones(shape=num_data) / num_data

        for boost_iter in range(self.iter):
            # Calculate errors for each candidate classifier
            step_error = list()
            self.clfrs = deepcopy(self.clfs) # Every iteration needs resetting of individual voters(state of not fitted)
            for i in self.clfrs:
                print(self.sample_weights[boost_iter])
                # Get the classifier that has minimum training error
                i.fit(X_train, y_train, sample_weight=self.sample_weights[boost_iter])
                pred = i.predict(X_train)
                ans = list(y_train[y_train.columns[0]]) # pd.DataFrame to List object

                # sum of sample weights when the prediction is wrong
                error = sum(c for a, b, c in zip(pred, ans, self.sample_weights[boost_iter]) if a != b)
                step_error.append(error)

            # Choose classifier that has minimum error
            minimum = min(step_error)
            if minimum > 0.5:
                raise ValueError("Accuracy of all the weak learner less than 0.5")

            min_ind = step_error.index(minimum)

            # Fitting the classifier that has miminum error
            # Classifiers are already fitted. Therefore there are no need to re-fit it with training data.
            pred = self.clfrs[min_ind].predict(X_train)
            ans = list(y_train[y_train.columns[0]])

            # Calculate classifier weight
            alpha_clf = (1/2) * np.log((1 - minimum)/minimum)

            # Calculate updated sample
            new_sample_weight = self.sample_weights[boost_iter] * np.exp(-1 * alpha_clf * np.array(pred) * np.array(ans))
            if boost_iter < self.iter:
                if boost_iter == (self.iter - 1): # Last iteration
                    pass
                else:
                    self.sample_weights[boost_iter + 1] = new_sample_weight

            # Results of an iteration
            self.classifier[boost_iter] = self.clfrs[min_ind] # classifiers "fitted" results are stored
            self.clf_weights[boost_iter] = alpha_clf
            self.errors[boost_iter] = min(step_error)

        return self

    def predict(self, X_test: pd.DataFrame):
        """
        Get total prediction of AdaBoost.
        :return:
        """
        self.individual = np.array([clf.predict(X_test) for clf in self.classifier])
        res_sign = np.sign(np.dot(self.clf_weights, self.individual))
        return res_sign

    def get_iter_prediction(self):
        """
        Returns the prediction result of each iteration.
        """
        if self.individual == list():
            raise AssertionError("predict function must be executed to get iteration prediction result")
        return self.individual

    def get_weights(self):
        return self.sample_weights

    def get_clf_weights(self):
        return self.clf_weights

    def get_iteration_errors(self):
        return self.errors

    def _early_stopping_rules(self):
        pass