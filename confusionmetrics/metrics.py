# Bismillahi-r-Rahmani-r-Rahim

import numpy as np

class MissingDataException(Exception):
    pass

def accuracy(confusion):
    """
    >>> accuracy([[1,0],[1,0]])
    0.5
    """
    return (np.sum(np.diagonal(confusion))/
            float(np.sum(confusion)))

def precision(confusion):
    """
    >>> precision([[5,1],[1,1]])
    0.5
    """
    judged_pos = confusion[0][1] + confusion[1][1]
    if judged_pos == 0:
        return 0.0
    return (confusion[1][1]/
            float(judged_pos))

def recall(confusion):
    """
    >>> recall([[5,1],[1,3]])
    0.75
    """
    pos = confusion[1][0] + confusion[1][1]
    if pos == 0:
        return 0.0
    return (confusion[1][1]/
            float(pos))

def f1_score(confusion):
    """
    >>> f1_score([[1,1],[1,1]])
    0.5
    """
    p = precision(confusion)
    r = recall(confusion)
    if p + r == 0.0:
        return 0.0
    return 2*p*r/(p + r)

def fbeta(confusion, beta):
    p = precision(confusion)
    r = recall(confusion)
    if p + r == 0.0:
        return 0.0    
    beta2 = beta ** 2
    fscore = (1 + beta2) * (p * r) / (
        beta2 * p + r)
    return fscore
