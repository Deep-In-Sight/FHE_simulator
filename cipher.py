import numpy as np

def key_hash(key):
    return hash(str(key*np.conj(key)))

def secret_key_match_encryptor_key(enc, sk):
    """키 pair를 conjigate로 만들자! 
    """
    return key_hash(sk) == key_hash(enc)
   

class Parameters():
    def __init__(self):
        pass
    
class CKKS_Parameters(Parameters):
    def __init__(self):
        """"""
        self.logp = None
        self.logq = None
        self.logN = None

class Ring():
    def __init__(self, seed=1234):
        """Determine the random state for key generation (but nothing else than that)
        
        """
        # random_number_generator
        self.rng = np.random.default_rng(seed)

class Context():
    def __init__(self, params:Parameters, ring:Ring):
        self.params = params
        self._ring = ring
        self.enc_key = None
        self.generate_encryption_key()
        pass
    
    def generate_encryption_key(self, n=10):
        """Fake"""
        rng = self._ring.rng
        # You need a double parenthesis!
        self.enc_key = rng.random((n)) + rng.random((n))*1j
    
    def generate_secret_key(self):
        """Fake"""
        if self.enc_key is None:
            raise ValueError
        else:
            return np.conj(self.enc_key) 
        

        
