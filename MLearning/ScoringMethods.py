import pandas as pd
import numpy as np


def True_Positive(truth, pred):
    """the number of positive cases that are correctly identified as positive
    truth == 1, pred == 1"""
    TP = 0
    for elements in range(len(truth)):
        if truth[elements] == pred[elements] and truth[elements] == 1:
            TP += 1
        else:
            pass
    return TP


def True_Negative(truth, pred):
    """the number of negative cases that are correctly labeled as negative
    truth == 0, pred == 0"""
    TN = 0
    for elements in range(len(truth)):
        if truth[elements] == pred[elements] and truth[elements] == 0:
            TN += 1
        else:
            pass
    return TN


def False_Positive(truth, pred):
    """the number of positive cases that are miss-labeled as negative
    truth == 1, pred == 0"""
    FP = 0
    for elements in range(len(truth)):
        if truth[elements] != pred[elements] and truth[elements] == 1:
            FP += 1
        else:
            pass
    return FP

def False_Negative(truth, pred):
    """the number of negative cases that are miss-labeled as positive"""
    FN = 0
    for elements in range(len(truth)):
        if truth[elements] != pred[elements] and truth[elements] == 0:
            FN += 1
        else:
            pass
    return FN

class scoring_model:
    def __init__(self, true_data, test_data):
        """Data is pandas core Series
        positive data has a value of 1, and negative data has a value of -1"""
        self.truth = true_data
        self.pred = test_data
        # Check if the length are same
        if len(self.truth) != len(self.pred):
            raise ValueError("Length of true data and test data doesn't match")
        else:
            pass

    def sensitivity(self):
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return self.sens

    def specificity(self):
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        self.spec = self.TN / (self.TN + self.FP)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)
        return self.spec

    def gmean(self):
        """geometric mean.
        indicates balance between classification performance.
        poor performance in prediction of the positive cases will lead to low G-mean
        even if the negative cases are correctly classified by the evaluated algorithm"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)

        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return np.sqrt(self.sens * self.spec)

    def LP(self):
        """likelihood ratio represents the ratio between
        the probability of predicting an example as positive when it's actually positive
        &
        the probability of predicting an example as positive when it's not actually positive"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return self.sens / (1 - self.spec)

    def LR(self):
        """negative likelihood ratio"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return (1 - self.sens) / self.spec

    def DP(self):
        """Discriminant power. Summarizes sensitivity and specificity
        DP value higher than 3 indicate that the algorithm distinguishes well between positive & negative"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return (np.sqrt(3)/np.pi)*(np.log(self.sens / (1 - self.sens))
                                  + np.log(self.spec / (1 - self.spec)))

    def Youden(self):
        """Youden's gamma index
        linear transformation of the mean sensitivity and specifity
        Higher value of youden's gamma indicates better ability of the algorithm to avoid misclassification"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return self.sens - (1 - self.spec)

    def BA(self):
        """Balanced accuracy metric is the average of sensitivity and specificity"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
        return (1/2)*(self.sens + self.spec)

    def WBA(self):
        """Weighted balanced accuracy"""
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)
        if (self.TN + self.FP) == 0:
            self.spec = np.nan
        else:
            self.spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            self.sens = np.nan
        else:
            self.sens = self.TP / (self.TP + self.FN)
            # ADD NEXT TIME
        return None

