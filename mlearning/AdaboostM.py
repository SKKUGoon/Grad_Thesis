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

class AdaboostClassifierES:
    def __init__(self, voting_clf: List, max_iter, early_stopping: bool):
        ...
        self.clfs = voting_clf
        self.iter = max_iter
        self.stochasticfit = False

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

        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

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
                # Get the classifier that has minimum training error
                i.fit(X_train_np, y_train_np, sample_weight=self.sample_weights[boost_iter])
                pred = i.predict(X_train_np)
                ans = list(y_train[y_train.columns[0]]) # pd.DataFrame to List object

                # sum of sample weights when the prediction is wrong
                error = sum(c for a, b, c in zip(pred, ans, self.sample_weights[boost_iter]) if a != b)
                if error == 0:
                    print(i, 'zero training error')
                    pass
                else:
                    step_error.append(error)

            # Choose classifier that has minimum error
            minimum = min(step_error)
            if minimum > 0.5:
                raise ValueError("Accuracy of all the weak learner less than 0.5")

            min_ind = step_error.index(minimum)

            # Fitting the classifier that has miminum error
            # Classifiers are already fitted. Therefore there are no need to re-fit it with training data.
            pred = self.clfrs[min_ind].predict(X_train_np)
            ans = list(y_train[y_train.columns[0]])

            # Calculate classifier weight
            alpha_clf = (1/2) * np.log((1 - minimum)/minimum)

            # Calculate updated sample
            new_sample_weight = self.sample_weights[boost_iter] * np.exp(-1 * alpha_clf * np.array(pred).transpose() * np.array(ans))

            if len(new_sample_weight) == 1: # Goonsik Debugging
                new_sample_weight = new_sample_weight[0]

            if boost_iter < self.iter:
                if boost_iter == (self.iter - 1): # Last iteration
                    pass
                else:
                    self.sample_weights[boost_iter + 1] = new_sample_weight / sum(new_sample_weight) # to make sum(weight) = 1

            # Results of an iteration
            self.classifier[boost_iter] = self.clfrs[min_ind] # classifiers "fitted" results are stored
            self.clf_weights[boost_iter] = alpha_clf
            self.errors[boost_iter] = min(step_error)

        return self

    def stochastic_fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        For m=1 to L
        1) Fit the classifier and get the best fit(smallest training error)
        2) Get the classifier's training error (e)
        3) Compute weight of the classifier
        weight = (1/2) * np.log((1-e)/e)
        4) set new sample weight and normalize it to 1
        """
        self._validity_check(X_train, y_train)

        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        self.stochasticfit = True

        # Get optimal classifier for the iteration
        num_feat = len(X_train.columns)
        num_data = len(X_train)

        # Number of iterations are needed to proceed with data
        self.sample_weights = np.zeros(shape=(self.iter, num_data))
        self.classifier = np.zeros(shape=self.iter, dtype=object)
        self.clf_weights = np.zeros(shape=self.iter)
        self.clf_choice_weights = np.zeros(shape=(self.iter, len(self.clfs)))
        self.errors = np.zeros(shape=self.iter)

        # Set initial sample weight(all equal)
        self.sample_weights[0] = np.ones(shape=num_data) / num_data

        # Set initial class_choice weight(all equal)
        self.clf_choice_weights[0] = np.ones(shape=len(self.clfs)) / len(self.clfs)

        for boost_iter in range(self.iter):
            # Calculate errors for each candidate classifier
            clfrs = deepcopy(self.clfs) # Every iteration needs resetting of individual voters(state of not fitted)
            # Choose classifier randomly
            clfrs = np.array(clfrs)
            ind = np.random.choice(len(self.clfs), p = self.clf_choice_weights[boost_iter])

            iterclf = clfrs[ind]
            iterclf.fit(X_train_np, y_train_np, sample_weight=self.sample_weights[boost_iter])
            pred = iterclf.predict(X_train_np)
            ans = list(y_train[y_train.columns[0]]) # pd.DataFrame to List object

            # sum of sample weights when the prediction is wrong
            error = sum(c for a, b, c in zip(pred, ans, self.sample_weights[boost_iter]) if a != b)
            if error == 0:
                print(iterclf, 'zero training error')
                return self

            # Calculate classifier weight
            alpha_clf = (1/2) * np.log((1 - error)/error)

            # Calculate updated sample
            new_sample_weight = self.sample_weights[boost_iter] * np.exp(-1 * alpha_clf * np.array(pred).transpose() * np.array(ans))

            if len(new_sample_weight) == 1:  # Goonsik Debugging
                new_sample_weight = new_sample_weight[0]

            if boost_iter < self.iter:
                if boost_iter == (self.iter - 1):  # Last iteration
                    pass
                else:
                    self.sample_weights[boost_iter + 1] = new_sample_weight / sum(new_sample_weight)  # to make sum(weight) = 1
            # Calculate updated classifier choice weight
            # find the location of the classifier & put the weight there

            new_clf_choice = deepcopy(self.clf_choice_weights[boost_iter])
            new_clf_choice[ind] = new_clf_choice[ind] * np.exp(-1 * alpha_clf)
            if boost_iter < self.iter:
                if boost_iter == (self.iter - 1):
                    pass
                else:
                    self.clf_choice_weights[boost_iter + 1] = new_clf_choice / sum(new_clf_choice) # to make sum(weight) = 1


            # Results of an iteration
            self.classifier[boost_iter] = iterclf # classifiers "fitted" results are stored
            self.clf_weights[boost_iter] = alpha_clf
            self.errors[boost_iter] = error
            print(f'{boost_iter+1} done')

        return self

    def predict(self, X_test: pd.DataFrame, raw_result=False):
        """
        Get total prediction of AdaBoost.
        :return:
        """
        X_test_np = X_test.to_numpy()
        self.individual = [clf.predict(X_test_np) for clf in self.classifier[self.classifier != 0]]
        idvdl = list()
        for pdt in self.individual: # Goonsik Debug
            if pdt.shape == (len(X_test),):
                idvdl.append(pdt)
            else:
                pdt = pdt.transpose()[0]
                idvdl.append(pdt)
        self.individual = np.array(idvdl)

        if raw_result is True:
            res = [w * prd for w, prd in zip(self.clf_weights, self.individual)]
            return sum(res)
        else:
            res = [w * prd for w, prd in zip(self.clf_weights, self.individual)]
            res_sign = np.sign(sum(res))
            return res_sign

    def get_iter_prediction(self):
        """
        Returns the prediction result of each iteration.
        """
        if self.individual == list():
            raise AssertionError("predict function must be executed to get iteration prediction result")

        return self.individual

    def get_iter_cumul_prediction(self):
        """
        Returns the prediction each iteration. In a cumulative manner
        """
        if self.individual == list():
            raise AssertionError("predict function must be executed to get iteration prediction result")

        res = list()
        for i in range(1, len(self.individual)):
            r = np.sign(np.dot(self.clf_weights[0:i], self.individual[0:i]))
            res.append(r)
        return res

    def get_classifier_choice_weight(self):
        if self.stochasticfit is True:
            return self.clf_choice_weights

    def get_iter_classifier(self):
        """
        Returns the classifier used for each iteration
        """
        return self.classifier

    def get_weights(self):
        """
        Returns sample weights of each iteration
        """
        return self.sample_weights

    def get_clf_weights(self):
        """
        Returns classifier weights for each iteration
        """
        return self.clf_weights

    def get_iter_errors(self):
        """
        Returns errors made by the classifier in each iterations.
        """
        return self.errors

    def _early_stopping_rules(self):
        pass