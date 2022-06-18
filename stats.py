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

    def variance(self, ctxt, ev):
        ev = self.evaluator
        algo = self.algorithms

        n = ctxt._n_elements
        summed = algo.sum_reduce(ctxt, partial=True, duplicate=True)
        summed = algo.put_mask(summed, np.arange(ctxt._n_elements))
        mean = ev.div_by_plain(summed, n)
        sub = ev.sub(ctxt, mean)
        squared = ev.mult(sub, sub, inplace=False)
        summed_sq = algo.sum_reduce(squared, partial=True, duplicate=False)
        return ev.div_by_plain(summed_sq, n)