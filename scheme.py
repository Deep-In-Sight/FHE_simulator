import numpy as np
from cipher import *
from ciphertext import *

class Encryptor():
    def __init__(self, enc_key):
        self._enc_key = enc_key
    
    def encrypt(self, arr):
        # How to determine if I want Ciphertext of CiphertextStat?
        ctxt = CiphertextStat(arr)
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
        if ctxt._enckey_hash == self._sk_hash:
            ctxt._encrypted = False
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

    #@compatibility_check_ctxt
    def add(self, ctxt1, ctxt2):
        if not compatible(ctxt1, ctxt2):
            raise 
        else:
            pass

    #@compatibility_check_ptxt


def compatible(ctxt1, ctxt2):
    """What else should I check? """
    return ctxt1.get_scale() == ctxt2.get_scale()
