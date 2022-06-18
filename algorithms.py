import numpy as np

from ciphertext import CiphertextStat
from scheme import Evaluator

# class Mask():
#     def __init__(self, beg, fin, stride):
#         """Flexible mask class
        
#         Need to devise a concise and effective pattern generation method
#         -- Like numpy Slice Tile tensor. 
#         """
#         self.beg = beg
#         self.fin = fin
#         self.stride = stride



class Algorithms():
    def __init__(self, evaluator:Evaluator):
        self.evaluator = evaluator

    def sum_reduce(self,
                    ctxt:CiphertextStat, 
                    partial=False, 
                    duplicate=False): 
        """calculate sum of all elements in the array.


        Use cases
        -----
        1. The sum of the array ends up at the first slot. 
            Some of other slots are contaminated. (may need masking later)
        2. The sum of the array fills valid slots.
            Some of other slots are contaminated. (may need masking later)
        3. The sum of the array fills all slots.
            No contaminated slots.


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
        if not partial and duplicate:
            raise ValueError("Partial = False, duplicate = True not allowed.")
        ev = self.evaluator

        if partial:
            n = ctxt._n_elements
        else:
            n = ctxt.nslots
        log2n = np.log2(n).astype(int)

        # keep the original ctxt intact
        ctxt_ = ev.copy(ctxt)
        if duplicate:
            # shifted copy
            rot_copy = ev.copy(ctxt)
            ev.lrot(rot_copy, -ctxt_._n_elements)

            ev.add(ctxt_, rot_copy, inplace=True)
        for i in range(log2n):
            tmp = ev.copy(ctxt_)
            ev.lrot(tmp, 2**i, inplace=True)
            ev.add(ctxt_, tmp, inplace=True)
        return ctxt_

    def put_mask(self, ctxt:CiphertextStat, mask:slice, inplace=False):
        """Multily ctxt by a plain mask

        Mask could evolve into much more flexible one. 
        to suppor convolutions, for example.
        """
        ev = self.evaluator

        _mask = np.zeros(ctxt.nslots)
        _mask[mask] = 1
        if inplace:
            ev.mult_by_plain(ctxt, _mask, inplace=True)
        else:
            return ev.mult_by_plain(ctxt, _mask, inplace=False)

    def sqrt(self, ctxt:CiphertextStat, inplace=False):
        """

        It's not that trivial...
        """

    def inv(self, ctxt:CiphertextStat, number:float=1e-4, n_iters=20):
        """Division by Newton-Raphson method.
        https://en.wikipedia.org/wiki/Division_algorithm#Newton%E2%80%93Raphson_division

        parameters
        ----------
        number: initial guess 0 < ig < 1
        Other methods may worth try.
        
        NOTE
        ----
        Needs an error analysis

        """
        ev = self.evaluator

        # In the first iteration, number is ptxt
        q_ = ev.mult_by_plain(ctxt, -number, inplace=False)
        sub_ = ev.add_plain(q_, 2, inplace=False)
        number_ = ev.mult_by_plain(sub_, number, inplace=False)

        for i in range(1, n_iters):
            tmp = ev.mult_by_plain(number_, -1, inplace=False)
            q_ = ev.mult(ctxt, tmp, inplace=False)
            sub_ = ev.add_plain(q_, 2, inplace=False)
            ev.mult(number_, sub_, inplace=True)

        return number_