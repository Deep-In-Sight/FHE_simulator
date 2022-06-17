import numpy as np

from ciphertext import CiphertextStat
from scheme import Evaluator

class Algorithms():
    def __init__(self, evaluator:Evaluator):
        self.evaluator = evaluator

    def fhe_sum(self,
                ctxt:CiphertextStat, 
                partial=False): 
        """calculate sum of all elements in the array.

        note
        ----
        1. When the ciphertext is not fully packed (when n_elements < nslots),
        rotating and summing only log2(n_elements) can reduce the cost
        whlie only the first slot ends up holding the complete summation.
        Depending on the downstream calculations, sum may need to be broadcasted accross slots.
        Unless n_elements << nslots, duplicating the first slot to other slots takes
        as many rotations as log2(#_valid_slots -1), plus a multiplication by a mask. 
        So, we let partial default to False.

        2. Error accumulation
        It performs log2(n_elements) rotations and additions.
        When n_elements = 50M, it is 26 times of consequtive summation. 
        Error accumulation may cause some trouble. 
        We need a robust way to monitor error growth.
        """
        if partial:
            n = ctxt._n_elements
        else:
            n = ctxt.nslots
        log2n = np.log2(n).astype(int)

        # keep the original ctxt intact
        ctxt_ = self.evaluator.copy(ctxt)
        for i in range(log2n):
            tmp = self.evaluator.copy(ctxt_)
            self.evaluator.lrot(tmp, 2**i, inplace=True)
            self.evaluator.add(ctxt_, tmp, inplace=True)
        
        return ctxt_