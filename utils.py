import numpy as np
import scipy.stats

#http://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    """
    Usage
    a = [1, 2, 3, 4, 5, 6]
    m, h  = mean_confidence_interval(a)
    print '%4.2f mean, %4.2f interval' % (m, h)

    """
    a = 1.0 * np.array(data)
    size = len(a)
    mean, std_err_mean = np.mean(a), scipy.stats.sem(a)
    h = std_err_mean * scipy.stats.t._ppf((1+confidence)/2., size-1)
    return mean, h



