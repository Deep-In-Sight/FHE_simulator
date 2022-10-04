import numpy as np
from numpy import polynomial as P
import math 

from ciphertext import Ciphertext, CiphertextStat, Plaintext
from scheme import Evaluator, Encoder

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
    def __init__(self, evaluator:Evaluator, encoder=Encoder):
        self.evaluator = evaluator
        self.encoder = encoder
        self._nslots = self.evaluator.context.params.nslots
        self._sign = self._approx_sign(10)

    def encode_repeat(self, v, logp=None):
        return self.encoder.encode(np.repeat(v, self._nslots),
                            logp=logp)

    def sum_reduce(self,
                    ctxt:CiphertextStat, 
                    nsum=None,
                    partial=False, 
                    duplicate=False): 
        """calculate sum of **nsum** elements in the array.
        => sum([0, nsum]), sum([1, nsum+1]), ...


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

        if not nsum:
            if partial:
                nsum = ctxt._n_elements
            else:
                nsum = ctxt.nslots
        log2n = np.log2(nsum).astype(int)

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
        _mask = self.encoder.encode(_mask)

        if inplace:
            ev.mult_by_plain(ctxt, _mask, inplace=True)
        else:
            return ev.mult_by_plain(ctxt, _mask, inplace=False)

    def inv(self, ctxt:Ciphertext, number:float=1e-4, n_iters=20):
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

    def divide(self, ctxt1:Ciphertext, ctxt2:Ciphertext,  inplace=False):
        """Fake implementation"""
        ev = self.evaluator
        if not inplace:
            new_ctxt = ev.copy(ctxt1)
        else:
            new_ctxt = ctxt1
        
        new_ctxt._arr /= ctxt2._arr

        return new_ctxt
        


################# SQRT #################
    def _inv_sqrt_initial_guess(self, n_iter, tol):
        """

        """
        pass

    def inv_sqrt(self, ctxt:CiphertextStat):
        """
        https://github.com/pandasamanvaya/Pivot-tangent
        Newton's method + pivot point
        """
        pass

    def sqrt(self, ctxt:CiphertextStat, inplace=False):
        ev = self.evaluator

        return ev.mult(ctxt, self.inv_sqrt(ctxt), inplace=False)

################# comp #################
    def sign(self, ctxt):
        new_ctxt = CiphertextStat(ctxt)

        #### TODO ####
        #poly_eval로 계산해야함!!!!!!!!!!!!!!!!!
        new_ctxt._arr = self._sign(ctxt._arr)
        return new_ctxt

    def sign_ranged(self, ctxt, range):
        """calculate sign of variables in a non-standard range.

        accounting for scaling factor in generating the fitting function is cheaper 
        than multiplying ctxt with a plaintext.
        """
        new_ctxt = CiphertextStat(ctxt)

        #### TODO ####
        #poly_eval로 계산해야함!!!!!!!!!!!!!!!!!
        new_ctxt._arr = self._sign(ctxt._arr)
        return new_ctxt

    def comp(self, ctxt1:Ciphertext, oper2):
        """sign 두 번 composite
        """
        if isinstance(oper2, Ciphertext):
            diff = self.evaluator.sub(ctxt1, oper2)
        elif isinstance(oper2, Plaintext):
            diff = self.evaluator.sub_plain(ctxt1, oper2)
        one = Plaintext(arr=np.repeat(1., diff.nslots),
                     logp = diff.logp, nslots = diff.nslots)
        half = Plaintext(arr=np.repeat(.5, diff.nslots),
                     logp = diff.logp, nslots = diff.nslots)
        sign_plus1 = self.evaluator.add_plain(self.sign(self.sign(diff)), one)
        return self.evaluator.mult_by_plain(sign_plus1, half)

    def _powerExtended(self, ctxt:CiphertextStat,degree:int):
        """
        SchemeAlgo::powerExtended(Ciphertext* res, Ciphertext& cipher, long logp, long degree)
        """

    def eval_poly(self, ctxt:CiphertextStat, coeff:list, tol=1e-6):
        pass

    @staticmethod
    def _approx_sign(n):
        """
        Approxiate sign function in [-1,1]
        """
        p_t1 = P.Polynomial([1,0,-1])
        p_x  = P.Polynomial([0,1])
        
        def c_(i: int):
            return 1/4**i * math.comb(2*i,i)

        def term_(i: int):
            return c_(i) * p_x * p_t1**i

        poly = term_(0)
        for nn in range(1,n+1):
            poly += term_(nn)
        return poly

    