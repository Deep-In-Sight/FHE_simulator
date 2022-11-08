from typing import Dict
import numpy as np
from .cipher import *
from .ciphertext import *
from .errors import *
from .utils import *
from copy import copy

class Encoder():
    def __init__(self, context):
        self.logp = context.params.logp
        self.nslots = context.params.nslots

    def encode(self, arr, logp=None, nslots=None):
        if logp:
            self.logp = logp
        if nslots:
            self.nstlos = nslots

        assert(self.logp != None), 'Ptxt scale not set'

        return Plaintext(arr=arr, logp = self.logp, nslots = self.nslots)


class Encryptor():
    def __init__(self, context:Context):
        self._context = context
        self._enc_key = self._context.enc_key
        self.enckey_hash = key_hash(self._enc_key)
    
    def encrypt(self, arr):
        # How to determine if I want Ciphertext or CiphertextStat?
        ctxt = CiphertextStat(self._context.params.logp, 
                              self._context.params.logq,
                              self._context.params.logn)
        
        #encoded = _stringify(arr)
        ctxt._set_arr(self.enckey_hash, arr)
        # TODO: Need to determine nslots 
        return ctxt

class Decryptor():
    def __init__(self, secret_key):
        self._secret_key = secret_key
        self._sk_hash = key_hash(secret_key)
    
    def decrypt(self, ctxt):
        if ctxt._enckey_hash == self._sk_hash and ctxt._encrypted:
            return ctxt._arr
        else:
            raise ValueError("You have a wrong secret key")

@staticmethod
def create_new_ctxt(ctxt):
    newctxt = CiphertextStat(logp = ctxt.logp,
                        logq = ctxt.logq,
                        logn = ctxt.logn)
    
    newctxt._enckey_hash = ctxt._enckey_hash
    return newctxt

@staticmethod
def copy_ctxt(ctxt:CiphertextStat):
    """copy a ciphertextStat instance"""
    new_ctxt = create_new_ctxt(ctxt)
    new_ctxt._set_arr(ctxt._enckey_hash, ctxt._arr, 
                        n_elements=ctxt._n_elements)
    new_ctxt._encrypted = ctxt._encrypted
    new_ctxt._n_elements = ctxt._n_elements
    return new_ctxt

class Evaluator():
    def __init__(self, keys:Dict, context:Context):
        self._multiplication_key = keys['mult']
        self.rotation_keys = keys['rot']
        self.multkey_hash = key_hash(-1*self._multiplication_key)
        self._logp = context.params.logp
        self.context = context
        self._counter = Call_counter()

    def copy(self, ctxt):
        return copy_ctxt(ctxt)

    def bootstrap(self, ctxt:Ciphertext):
        ctxt.logq = self.context.logq - ctxt.logp
        self._counter.bootstrap(ctxt)

    def negate(self, ctxt:Ciphertext, inplace=True):
        if inplace:
            ctxt._arr = -1*ctxt._arr
            return ctxt
        else:
            new_ctxt = self.copy(ctxt)
            new_ctxt._arr = -1*new_ctxt._arr
            return new_ctxt

    def _change_mod(self, ctxt:Ciphertext, logq):
        """
        proxy for Scheme.change_mod
        """
        ctxt.logq = logq
        self._counter.mod_switch(ctxt)

    def mod_down_by(self, ctxt:Ciphertext, logp, inplace=True):
        assert ctxt.logq > logp, "Cannot mod down any further"
        if inplace:
            self._change_mod(ctxt, ctxt.logq - logp)
        else:
            new_ctxt = self.copy(ctxt)
            self._change_mod(new_ctxt, ctxt.logq - logp)
            return new_ctxt

    def mod_down_to(self, ctxt:Ciphertext, logq):
        assert ctxt.logq >= logq, "Cannot mod down to a higher level"
        self._change_mod(ctxt, logq)

    def match_mod(self, ctxt1:Ciphertext, ctxt2:Ciphertext):
        self._change_mod(ctxt2, min([ctxt1.logq, ctxt2.logq]))
        self._change_mod(ctxt1, ctxt2.logq)

    @staticmethod
    @check_compatible
    def _add(ctxt1:Ciphertext, ctxt2:Ciphertext):
        """proxy for HEAAN.ring.add() and ring.addAndEqual()
        """
        return ctxt1._arr + ctxt2._arr
        
    def add(self, ctxt1:CiphertextStat, ctxt2:CiphertextStat, inplace=False):
        assert self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash, "Eval key and Enc keys don't match"
        if inplace:
            ctxt1._arr = self._add(ctxt1,ctxt2)
        else:
            new_ctxt = create_new_ctxt(ctxt1)
            new_ctxt._set_arr(ctxt1._enckey_hash, self._add(ctxt1,ctxt2))
            return new_ctxt

    @staticmethod
    @check_compatible
    def _add_plain(ctxt:Ciphertext, ptxt:Plaintext):
        """proxy for HEAAN.ring.addConst() and ring.addConstAndEqual()
        """
        return ctxt._arr + ptxt._arr
        
    def add_plain(self, ctxt:CiphertextStat, ptxt:Plaintext, logp=None, inplace=False):
        #assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc keys don't match"
        if not isinstance(ptxt, Plaintext) and logp is not None:
            ptxt = Plaintext(arr=np.repeat(ptxt, ctxt.nslots), logn=ctxt.logn, logp=logp)
        if inplace:
            ctxt._arr = self._add_plain(ctxt,ptxt)
        else:
            new_ctxt = create_new_ctxt(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._add_plain(ctxt,ptxt))
            return new_ctxt

    @staticmethod
    @check_compatible
    def _sub(ctxt1:Ciphertext, ctxt2:Ciphertext):
        """proxy for HEAAN.ring.sub() and ring.subAndEqual1,2()
        """
        return ctxt1._arr - ctxt2._arr
        
    def sub(self, ctxt1:CiphertextStat, ctxt2:CiphertextStat, inplace=False):
        assert self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash, "Eval key and Enc keys don't match"
        if inplace:
            ctxt1._arr = self._sub(ctxt1,ctxt2)
        else:
            new_ctxt = create_new_ctxt(ctxt1)
            new_ctxt._set_arr(ctxt1._enckey_hash, self._sub(ctxt1,ctxt2))
            return new_ctxt

    @staticmethod
    @check_compatible
    def _sub_plain(ctxt:Ciphertext, ptxt:Plaintext):
        """proxy for HEAAN.ring.sub() and ring.subAndEqual1,2()
        """
        return ctxt._arr - ptxt._arr
        
    def sub_plain(self, ctxt:CiphertextStat, ptxt:Plaintext, inplace=False):
        #assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc key don't match"
        if inplace:
            ctxt._arr = self._sub_plain(ctxt, ptxt)
        else:
            new_ctxt = create_new_ctxt(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._sub_plain(ctxt, ptxt))
            return new_ctxt
        

    def _mult(self, ctxt1:Ciphertext, ctxt2:Ciphertext):
        """
        """
        if not ctxt1._ntt:
            self.switch_ntt(ctxt1)
        if not ctxt2._ntt:
            self.switch_ntt(ctxt2)        

        self._counter.multc(ctxt1)
        return ctxt1._arr * ctxt2._arr
        
    def mult(self, ctxt1, ctxt2, inplace=False):
        assert self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash, "Eval key and Enc keys don't match"        
        if inplace:
            ctxt1._arr = self._mult(ctxt1,ctxt2)
            ctxt1.logp += ctxt2.logp
            
        else:
            new_ctxt = create_new_ctxt(ctxt1)
            new_ctxt._set_arr(ctxt1._enckey_hash, self._mult(ctxt1,ctxt2))
            new_ctxt.logp = ctxt1.logp + ctxt2.logp
            return new_ctxt

    def _mult_by_plain(self, ctxt, ptxt):
        self._counter.multp(ctxt)
        return ctxt._arr * ptxt._arr

    def mult_by_plain(self, ctxt:CiphertextStat, ptxt:Plaintext, logp=None, inplace=False):
        #assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc keys don't match"        
        if not isinstance(ptxt, Plaintext) and logp is not None:
            ptxt = Plaintext(arr=np.repeat(ptxt, ctxt.nslots), logn=ctxt.logn, logp=logp)
        if inplace:
            ctxt._arr = self._mult_by_plain(ctxt, ptxt)
            ctxt.logp += ptxt.logp            
        else:
            new_ctxt = create_new_ctxt(ctxt)
            new_ctxt._arr = self._mult_by_plain(ctxt, ptxt)
            new_ctxt.logp = ctxt.logp + ptxt.logp
            return new_ctxt

    def _square(self, ctxt:Ciphertext):
        """
        proxy for Scheme.square
        """
        if not ctxt._ntt:
            self.switch_ntt(ctxt)

        self._counter.multc(ctxt)
        return ctxt._arr**2
        
    def square(self, ctxt, inplace=False):
        assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc key don't match"        
        if inplace:
            ctxt._arr = self._square(ctxt)
            ctxt.logp *=2
        else:
            new_ctxt = create_new_ctxt(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._square(ctxt))
            new_ctxt.logp = ctxt.logp * 2
            return new_ctxt

    def _leftrot(self, ctxt:Ciphertext, r:int):
        """
        """
        self._counter.rot(ctxt)
        return np.roll(ctxt._arr, -r)
    
    def switch_ntt(self, ctxt:Ciphertext):
        ctxt._ntt = not ctxt._ntt
        self._counter.ntt_switch(ctxt)
        
    def lrot(self, ctxt:CiphertextStat, r:int, inplace=True):
        """Left-rotate ciphertext.

        note
        ----
        Right-rotation implemented in HEAAN and SEAL is simply 
        lrot(nslots - r). 
        Thus, rightrot is merely a convenience function.
        """
        assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc key don't match"
        
        # TODO: assert rotation_key exists
        if ctxt._ntt:
            self.switch_ntt(ctxt)

        if inplace:
            ctxt._arr = self._leftrot(ctxt, r)
        else:
            new_ctxt = create_new_ctxt(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._leftrot(ctxt, r))
            return new_ctxt

    def div_by_plain(self, ctxt, ptxt, inplace=False):
        inv_ptxt = copy(ptxt)
        inv_ptxt._arr = 1./inv_ptxt._arr
        if inplace:
            self.mult_by_plain(ctxt, inv_ptxt, inplace=inplace)
        else:
            return self.mult_by_plain(ctxt, inv_ptxt, inplace=inplace)
    
    def _reduce_logq(self, ctxt, delta):
        ctxt.logq -= delta
        assert ctxt.logq > 0, "no more noise budget! do bootstrapping"
        self._counter.rescale(ctxt)

    def rescale_next(self, ctxt:Ciphertext):
        """lower ctxt's scale by default scale"""
        delta = self._logp
        ctxt.logp -= delta
        self._reduce_logq(ctxt, delta)

    def rescale_to(self, ctxt1:Ciphertext, ctxt2:Ciphertext):
        assert ctxt1.logp > ctxt2.logp, "can't raise ctxt's scale"
        delta = ctxt1.logp - ctxt2.logp
        ctxt1.logp -= delta
        self._reduce_logq(ctxt1, delta)        
    
    def rescale_by(self, ctxt:Ciphertext, delta):
        assert ctxt.logp > delta, "can't raise ctxt's scale"
        ctxt.logp -= delta
        self._reduce_logq(ctxt, delta)


    def powerOf2Extended(self, ctxt:Ciphertext, logp, logDegree):
        res = [self.copy(ctxt)]
        for i in range(1, logDegree+1):
            res.append(self.square(res[-1]))
            self.rescale_by(res[-1], logp)
        
        return res

    def powerExtended(self, ctxt:Ciphertext, logp, degree):
        logDegree = np.log2(degree).astype(int)
        cpows = self.powerOf2Extended(ctxt, logp, logDegree)
        
        res = []
        for i in range(logDegree):
            powi = 1 << i
            res.append(self.copy(cpows[i]))
            for j in range(powi-1):
                bitsDown = res[j].logq - cpows[i].logq
                res.append(self.mod_down_by(res[j], bitsDown, inplace=False))
                self.mult(res[-1], cpows[i], inplace=True)
                self.rescale_by(res[-1], logp)
                
        res.append(self.copy(cpows[logDegree]))
        degree2 = 1 << logDegree
        for i in range(degree - degree2):
            bitsDown = res[i].logq - cpows[logDegree].logq
            res.append(self.mod_down_by(res[i], bitsDown, inplace=False))
            self.mult(res[-1], cpows[logDegree], inplace=True)
            self.rescale_by(res[-1], logp)
        
        return res
            
    def function_poly(self, ctxt, coeffs, logp):
        degree = len(coeffs) - 1
        cpows = self.powerExtended(ctxt, logp, degree)
        dlogp = 2*logp
        res = self.mult_by_plain(cpows[0], coeffs[1], logp, inplace=False)
        self.add_plain(res, coeffs[0], dlogp, inplace=True)
        
        for i in range(1, degree):
            if abs(coeffs[i+1]) > 1e-20:
                aixi = self.mult_by_plain(cpows[i], coeffs[i+1], logp)
                self.mod_down_to(res, aixi.logq)
                self.add(res, aixi, inplace=True)
        
        self.rescale_by(res, logp)
        return res 