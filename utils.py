from ciphertext import CiphertextStat
from errors import ScaleMisMatchError
import numpy as np
"""
HEAAN's Ciphertext class has two construtors.
1: 	Ciphertext(long logp = 0, long logq = 0, long n = 0);
2:	Ciphertext(const Ciphertext& o);

"""

def key_hash(key):
    return hash(str(key*np.conj(key)))

def secret_key_match_encryptor_key(enc, sk):
    """키 pair를 conjigate로 만들자! 
    """
    return key_hash(sk) == key_hash(enc)
   
def check_compatible(func):
    def func_wrapper(*args, **kwargs):
        ctxt1, ctxt2 = args
        if ctxt1.logp == ctxt2.logp:
            return func(*args, **kwargs)
        else:
            raise ScaleMisMatchError
    return func_wrapper

