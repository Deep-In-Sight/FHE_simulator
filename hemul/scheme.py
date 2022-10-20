from typing import Dict
import numpy as np
from .cipher import *
from .ciphertext import *
from .errors import *
from .utils import *
from copy import copy

class Encryptor():
    def __init__(self, context:Context):
        self._context = context
        self._enc_key = self._context.enc_key
        self.enckey_hash = key_hash(self._enc_key)
    
    def encrypt(self, arr):
        # How to determine if I want Ciphertext of CiphertextStat?
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
            
class Evaluator():
    def __init__(self, keys:Dict, context:Context):
        self._multiplication_key = keys['mult']
        self.rotation_keys = keys['rot']
        self.multkey_hash = key_hash(-1*self._multiplication_key)
        self._logp = context.params.logp
        self.context = context

    def bootstrap(self, ctxt:Ciphertext):
        ctxt.logq = self.context.logq - ctxt.logp

    def negate(self, ctxt:Ciphertext, inplace=True):
        if inplace:
            ctxt._arr = -1*ctxt._arr
            return ctxt
        else:
            new_ctxt = self.copy(ctxt)
            new_ctxt._arr = -1*new_ctxt._arr
            return new_ctxt

    @staticmethod
    def mod_down_by(ctxt:Ciphertext, logp):
        assert ctxt.logp > logp, "Cannot mod down any further"
        ctxt.logq -= logp

    @staticmethod
    def mod_down_to(ctxt:Ciphertext, logq):
        assert ctxt.logq >= logq, "Cannot mod down to a higher level"
        ctxt.logq = logq

    @staticmethod
    def match_mod(ctxt1:Ciphertext, ctxt2:Ciphertext):
        ctxt2.logq = min([ctxt1.logq, ctxt2.logq])
        ctxt1.logq = ctxt2.logq

    @staticmethod
    def copy(ctxt:CiphertextStat):
        """copy a ciphertextStat instance"""
        new_ctxt = CiphertextStat(ctxt)
        new_ctxt._set_arr(ctxt._enckey_hash, ctxt._arr, 
                          n_elements=ctxt._n_elements)
        return new_ctxt

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
            new_ctxt = CiphertextStat(ctxt1)
            new_ctxt._set_arr(ctxt1._enckey_hash, self._add(ctxt1,ctxt2))
            return new_ctxt

    @staticmethod
    @check_compatible
    def _add_plain(ctxt:Ciphertext, ptxt:Plaintext):
        """proxy for HEAAN.ring.addConst() and ring.addConstAndEqual()
        """
        return ctxt._arr + ptxt._arr
        
    def add_plain(self, ctxt:CiphertextStat, ptxt:Plaintext, inplace=False):
        #assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc keys don't match"
        if inplace:
            ctxt._arr = self._add_plain(ctxt,ptxt)
        else:
            new_ctxt = CiphertextStat(ctxt)
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
            new_ctxt = CiphertextStat(ctxt1)
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
            new_ctxt = CiphertextStat(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._sub_plain(ctxt, ptxt))
            return new_ctxt
        

    @staticmethod
    @check_compatible
    def _mult(ctxt1:Ciphertext, ctxt2:Ciphertext):
        """
        """
        return ctxt1._arr * ctxt2._arr
        
    def mult(self, ctxt1, ctxt2, inplace=False):
        assert self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash, "Eval key and Enc keys don't match"        
        if inplace:
            ctxt1._arr = self._mult(ctxt1,ctxt2)
            ctxt1.logp += ctxt2.logp
        else:
            new_ctxt = CiphertextStat(ctxt1)
            new_ctxt._set_arr(ctxt1._enckey_hash, self._mult(ctxt1,ctxt2))
            new_ctxt.logp = ctxt1.logp + ctxt2.logp
            return new_ctxt

    @staticmethod
    @check_compatible
    def _mult_by_plain(ctxt, ptxt):
        return ctxt._arr * ptxt._arr

    def mult_by_plain(self, ctxt:Ciphertext, ptxt:Plaintext, inplace=False):
        #assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc keys don't match"        
        if inplace:
            ctxt._arr = self._mult_by_plain(ctxt, ptxt)
            ctxt.logp += ptxt.logp
        else:
            new_ctxt = CiphertextStat(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._mult_by_plain(ctxt, ptxt))
            new_ctxt.logp = ctxt.logp + ptxt.logp
            return new_ctxt

    @staticmethod
    def _square(ctxt:Ciphertext):
        """
        proxy for Scheme.square
        """
        return ctxt._arr**2
        
    def square(self, ctxt, inplace=False):
        assert self.multkey_hash == ctxt._enckey_hash, "Eval key and Enc key don't match"        
        if inplace:
            ctxt._arr = self._square(ctxt)
            ctxt.logp *=2
        else:
            new_ctxt = CiphertextStat(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._square(ctxt))
            new_ctxt.logp = ctxt.logp * 2
            return new_ctxt


    @staticmethod
    def _leftrot(ctxt:Ciphertext, r:int):
        """
        """
        return np.roll(ctxt._arr, -r)
        
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

        if inplace:
            ctxt._arr = self._leftrot(ctxt, r)
        else:
            new_ctxt = CiphertextStat(ctxt)
            new_ctxt._set_arr(ctxt._enckey_hash, self._leftrot(ctxt, r))
            return new_ctxt

    def div_by_plain(self, ctxt, ptxt, inplace=False):
        inv_ptxt = copy(ptxt)
        inv_ptxt._arr = 1./inv_ptxt._arr
        if inplace:
            self.mult_by_plain(ctxt, inv_ptxt, inplace=inplace)
        else:
            return self.mult_by_plain(ctxt, inv_ptxt, inplace=inplace)
    @staticmethod
    def _reduce_logq(ctxt, delta):
        ctxt.logq -= delta
        assert ctxt.logq > 0, "no more noise budget! do bootstrapping"

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
    
def _stringify(arr):
    """convert array elements to a string"""
    return [str(a) for a in arr]

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



