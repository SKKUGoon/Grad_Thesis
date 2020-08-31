import pandas as pd
import numpy as np
from typing import List

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
        if truth[elements] == pred[elements] and truth[elements] == -1:
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
        if truth[elements] != pred[elements] and truth[elements] == -1:
            FN += 1
        else:
            pass
    return FN

class scoring_model:
    def __init__(self, true_data: List, test_data: List):
        """Data is pandas core Series
        positive data has a value of 1, and negative data has a value of -1"""
        self.truth = true_data
        self.pred = test_data

        # Check if the length are same
        if len(self.truth) != len(self.pred):
            raise ValueError("Length of true data and test data doesn't match")
        else:
            pass

        # Metrics needed
        self.TP = True_Positive(self.truth, self.pred)
        self.TN = True_Negative(self.truth, self.pred)
        self.FP = False_Positive(self.truth, self.pred)
        self.FN = False_Negative(self.truth, self.pred)


    def accuracy(self, weight=None) -> float:
        if weight is None:
            accy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
            return accy

        else:
            t = list(self.truth)

            t_c = list() # truth set's ac
            for i in sorted(list(set(t))):
                t_c.append(t.count(i))

            weight1 = t_c[1] # total(1): assign for every true -1
            weight2 = t_c[0] # total(-1): assign for every true 1

            accy = (weight2 * self.TP + weight1 * self.TN) / (2 * ((self.TP + self.FP) * (self.TN + self.FN)))
            return accy

    def sensitivity(self) -> float:
        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return sens


    def specificity(self) -> float:
        if (self.TN + self.FP) == 0:
            spec = np.inf
        else:
            spec = self.TN / (self.TN + self.FP)
        return spec


    def gmean(self) -> float:
        """geometric mean.
        indicates balance between classification performance.
        poor performance in prediction of the positive cases will lead to low G-mean
        even if the negative cases are correctly classified by the evaluated algorithm"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return np.sqrt(sens * spec)


    def LP(self) -> float:
        """likelihood ratio represents the ratio between
        the probability of predicting an example as positive when it's actually positive
        &
        the probability of predicting an example as positive when it's not actually positive"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return sens / (1 - spec)


    def LR(self) -> float:
        """negative likelihood ratio"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return (1 - sens) / spec


    def DP(self) -> float:
        """Discriminant power. Summarizes sensitivity and specificity
        DP value higher than 3 indicate that the algorithm distinguishes well between positive & negative"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return (np.sqrt(3)/np.pi) * (np.log(sens / (1 - sens)) + np.log(spec / (1 - spec)))


    def Youden(self) -> float:
        """Youden's gamma index
        linear transformation of the mean sensitivity and specifity
        Higher value of youden's gamma indicates better ability of the algorithm to avoid misclassification"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return sens - (1 - spec)


    def BA(self) -> float:
        """Balanced accuracy metric is the average of sensitivity and specificity"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
        return (1/2) * (sens + spec)


    def WBA(self) -> float:
        """Weighted balanced accuracy"""
        if (self.TN + self.FP) == 0:
            spec = np.nan
        else:
            spec = self.TN / (self.TN + self.FP)

        if (self.TP + self.FN) == 0:
            sens = np.nan
        else:
            sens = self.TP / (self.TP + self.FN)
            # ADD NEXT TIME
        return None

