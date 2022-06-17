import numpy as np
from cipher import *
from ciphertext import *
from errors import *


def check_compatible(func):
    def func_wrapper(*args, **kwargs):
        ctxt1, ctxt2 = args
        if ctxt1.logp == ctxt2.logp:
            return func(*args, **kwargs)
        else:
            raise ScaleMisMatchError
    return func_wrapper

class Encryptor():
    def __init__(self, context):
        self._context = context
        self._enc_key = self._context.enc_key
    
    def encrypt(self, arr):
        # How to determine if I want Ciphertext of CiphertextStat?
        ctxt = CiphertextStat(self._context.params.logp, 
                              self._context.params.logq,
                              self._context.params.nslots)
        ctxt._set_arr(arr)
        # TODO: Need to determine nslots 
        self._encrypt(ctxt)
        return ctxt
    
    def _encrypt(self, ctxt):
        ctxt._encrypt(self._enc_key)
    
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
    def __init__(self, keys):
        self.multiplication_key = keys['mult']
        self.rotation_keys = keys['rot']
    
    def rotate_left(self, ctxt, r):
        if self._rotation_key_exists(r):
            return np.roll(ctxt, -r)
        else:
            raise ValueError 
    
    def _rotation_key_exists(self, r):
        return self.rotation_keys[r] is not None

    @staticmethod
    @check_compatible
    def _add(ctxt1, ctxt2):
        """
        """
        return ctxt1._arr + ctxt2._arr
        

    def add(self, ctxt1, ctxt2, inplace=False):
        if inplace:
            ctxt1._arr = self._add(ctxt1,ctxt2)
        else:
            new_ctxt = CiphertextStat(ctxt1)
            new_ctxt._arr = self._add(ctxt1,ctxt2)
            return new_ctxt

    #@compatibility_check_ptxt


