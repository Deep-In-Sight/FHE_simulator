import numpy as np
from cipher import *
from ciphertext import *
from errors import *
from utils import *

class Encryptor():
    def __init__(self, context):
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
    
def _stringify(arr):
    """convert array elements to a string"""
    return [str(a) for a in arr]

def Encoder():
    def __init__(self):
        pass

    def encode(self, arr):
        return self._stringify(arr)

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
        self._multiplication_key = keys['mult']
        self.rotation_keys = keys['rot']
        self.multkey_hash = key_hash(-1*self._multiplication_key)
    
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
        if self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash:
            if inplace:
                ctxt1._arr = self._add(ctxt1,ctxt2)
            else:
                new_ctxt = CiphertextStat(ctxt1)
                new_ctxt._set_arr(ctxt1._enckey_hash, self._add(ctxt1,ctxt2))
                return new_ctxt
        else:
            print("Keys don't match")

    @staticmethod
    @check_compatible
    def _mult(ctxt1, ctxt2):
        """
        """
        return ctxt1._arr * ctxt2._arr
        
    def mult(self, ctxt1, ctxt2, inplace=False):
        if self.multkey_hash == ctxt1._enckey_hash == ctxt2._enckey_hash:
            if inplace:
                ctxt1._arr = self._mult(ctxt1,ctxt2)
            else:
                new_ctxt = CiphertextStat(ctxt1)
                new_ctxt._set_arr(ctxt1._enckey_hash, self._mult(ctxt1,ctxt2))
                return new_ctxt
        else:
            print("Keys don't match")


