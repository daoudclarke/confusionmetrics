# Bismillahi-r-Rahmani-r-Rahim

import unittest
import logging
import numpy as np
from numpy import random
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import fbeta_score, f1_score, confusion_matrix

from confusionmetrics import metrics

from learncone.ConeEstimatorBase import ConeEstimatorBase

class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testAccuracy(self):
        for i in range(2,100):
            true_values, judgments, confusion = self.get_test_values(i)
            
            expected = accuracy_score(true_values, judgments)
            actual = metrics.accuracy(confusion)
            
            self.assertEqual(expected, actual)

    def testPrecision(self):
        for i in range(2,100):
            true_values, judgments, confusion = self.get_test_values(i)
            
            expected = precision_score(true_values, judgments)
            actual = metrics.precision(confusion)
            
            self.assertEqual(expected, actual)

    def testRecall(self):
        for i in range(2,100):
            true_values, judgments, confusion = self.get_test_values(i)
            
            expected = recall_score(true_values, judgments)
            actual = metrics.recall(confusion)
            
            self.assertEqual(expected, actual)

    def testF1(self):
        for i in range(2,100):
            true_values, judgments, confusion = self.get_test_values(i)
            
            expected = f1_score(true_values, judgments)
            actual = metrics.f1_score(confusion)
            
            self.assertEqual(expected, actual)

    def testFBeta(self):
        for i in range(2,100):
            beta = i/10.0
            true_values, judgments, confusion = self.get_test_values(i)
            
            expected = fbeta_score(true_values, judgments, beta)
            actual = metrics.fbeta(confusion, beta)
            
            self.assertEqual(expected, actual)


    def get_test_values(self, i):
            true_values = np.array([x % 2 for x in range(i)])
            judgments = random.random_integers(0,1,i)
            confusion = confusion_matrix(true_values, judgments)
            return true_values, judgments, confusion


