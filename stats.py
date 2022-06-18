import numpy as np
from ciphertext import CiphertextStat
from scheme import Evaluator
from algorithms import Algorithms

class Statistics():
    def __init__(self, algorithms:Algorithms, evaluator:Evaluator):
        """Statistics module
        
        
        Consider keeping track of intermediate values
        """
        self.evaluator = evaluator
        self.algorithms = algorithms

    def variance(self, ctxt):
        """

        Check for further improvement:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        notably, there're online versions (which don't apply to SIMD setting),
        and paralle/distributed versions (very probably relevant to us)
        """
        ev = self.evaluator
        algo = self.algorithms

        n = ctxt._n_elements
        mean = self.mean(ctxt, partial=True, duplicate=True)
        mean = algo.put_mask(mean, np.arange(ctxt._n_elements))
        sub = ev.sub(ctxt, mean)
        squared = ev.mult(sub, sub, inplace=False)
        summed_sq = algo.sum_reduce(squared, partial=True, duplicate=False)
        return ev.div_by_plain(summed_sq, n)

    def std(self, ctxt):
        algo = self.algorithms

        return algo.sqrt(self.variance(ctxt))

    def stderr(self, ctxt):
        algo = self.algorithms
        ev = self.evaluator

        return ev.div_by_plain(self.std(ctxt),np.sqrt(ctxt._n_elements))

    def mean(self, ctxt, partial=True, duplicate=True):
        """

        example
        -------
        N = 1024
        _n_elements = 8   ([1,2,3,4,5,6,7,8])

        1. if partial = True, duplicate = True
        >>> print(mean[-8:])
        >>> [0.  1.  3.  6. 10. 15. 21. 28.] 
        >>> print(mean[:8])
        >>> [36. 36. 36. 36. 36. 36. 36. 36.]

        2. if partial = True, duplicate = False
        >>> print(mean[-8:])
        >>> [0.  1.  3.  6. 10. 15. 21. 28.] 
        >>> print(mean[:8])
        >>> [36. 35. 33. 30. 26. 21. 15.  8.]

        3. if partial = True, duplicate = False
        >>> print(mean[-8:])
        >>> [36. 36. 36. 36. 36. 36. 36. 36.] 
        >>> print(mean[:8])
        >>> [36. 36. 36. 36. 36. 36. 36. 36.]


        4. if partial = False, duplicate = True
        NOT ALLOWED 
        (will return [72. 72. 72. 72. 72. 72. 72. 72.] if allowed.)
        """
        ev = self.evaluator
        algo = self.algorithms

        n = ctxt._n_elements
        summed = algo.sum_reduce(ctxt, partial=partial, duplicate=duplicate)
        return ev.div_by_plain(summed, n)
        

    def cov(self, ctxt1, ctxt2):
        """

        Check for further improvement:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_covariance

        notably, there're online versions (which don't apply to SIMD setting),
        and paralle/distributed versions (very probably relevant to us)
        """
        ev = self.evaluator
        algo = self.algorithms

        n = ctxt1._n_elements -1       
        mean1 = self.mean(ctxt1, partial=True, duplicate=True)
        mean1 = algo.put_mask(mean1, np.arange(ctxt1._n_elements))
        
        mean2 = self.mean(ctxt2, partial=True, duplicate=True)
        mean2 = algo.put_mask(mean2, np.arange(ctxt1._n_elements))
        
        sub1 = ev.sub(ctxt1, mean1)
        sub2 = ev.sub(ctxt2, mean2)
        
        xy = ev.mult(sub1, sub2, inplace=False)
        summed_sq = algo.sum_reduce(xy, partial=True, duplicate=False)
        return ev.div_by_plain(summed_sq, n)

    def corrcoef(self, ctxt1, ctxt2):
        ev = self.evaluator
        algo = self.algorithms

        n = ctxt1._n_elements        
        
        mean1 = self.mean(ctxt1, partial=True, duplicate=True)
        mean1 = algo.put_mask(mean1, np.arange(ctxt1._n_elements))
        
        mean2 = self.mean(ctxt2, partial=True, duplicate=True)
        mean2 = algo.put_mask(mean2, np.arange(ctxt1._n_elements))
        
        sub1 = ev.sub(ctxt1, mean1)
        sub2 = ev.sub(ctxt2, mean2)
        
        xy = ev.mult(sub1, sub2, inplace=False)
        
        squared1 = ev.mult()

        xy = ev.mult(sub1, sub2, inplace=False)
        


        pass

    def coef_var(self,ctxt):
        ev = self.evaluator
        algo = self.algorithms
        
        return ev.mult(self.variance(ctxt), algo.inv(self.mean(ctxt)))