import numpy as np

def secret_key_match_encryptor_key(enc, sk):
    """키 pair를 conjigate로 만들자! 
    """
    return _key_hash(sk) == _key_hash(enc)
    
def _key_hash(key):
    return hash(str(key*np.conj(key)))

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
        

class Ciphertext():
    def __init__(self):
        """Base class for any ciphertext
        Currently following the CKKS convention -> should be changed.
        """
        self.nslots = None
        self.logp = None
        self.logq = None
        self.level = 0
        pass

class CiphertextStat(Ciphertext):
    def __init__(self, arr):
        """class

        min, max: 
            some algorithms utilize invariance to shift to enhance stability during calculation.
            e.g., var(X) = var(X-K) 
        In this case, the shift K needs to stay within the range of the array.

        mean:
            mean is probably the most popular quantity in statistical computations.

        n_lements:
            the length of the array, N, is a prerequisite to 
            calculating virtually all statistical quantities.
            We assume that this value is given. 
        """
        self._min = None  
        self._max = None
        self._mean = None
        self._arr = arr
        self._n_elements = self._arr.__len__()
        self._encrypted = False
        self._enckey_hash = None
    
    def _set_arr(self, arr):
        if not isinstance(arr, np.ndarray) or not np.issubdtype(arr, np.number):
            try:
                arr = np.array(arr)
            except:
                print("Need numeric np.ndarray")


    def _encrypt(self, encrypt_key):
        """to be called by Encryptor
        """
        self._enckey_hash = _key_hash(encrypt_key)
        self._encrypted = True
        
    def __repr__(self):
        if self._encrypted:
            return("You can't read the content")
        else:
            return self._arr.__repr__()
        

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
        self._sk_hash = _key_hash(secret_key)
    
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
        if self.rotation_keys[r] is not None:
            return True
        else:
            False
