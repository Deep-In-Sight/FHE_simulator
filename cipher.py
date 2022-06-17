import numpy as np

class Parameters():
    def __init__(self, logp, logq, logn):
        self.logp = logp
        self.logq = logq
        self.logn = logn
        self.nslots = int(2**self.logn)

    def __repr__(self):
        return f" logp: {self.logp}\n logq: {self.logq} \n logn: {self.logn}"
    
class CKKS_Parameters(Parameters):
    def __init__(self):
        """"""
        self.logp = None
        self.logq = None
        self.logn = None

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
        
    def generate_mult_key(self):
        """Fake"""
        if self.enc_key is None:
            raise ValueError
        else:
            return -1*np.conj(self.enc_key) 

