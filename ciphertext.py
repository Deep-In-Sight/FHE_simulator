import numpy as np
from cipher import key_hash

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
        super().__init__()
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
        self._enckey_hash = key_hash(encrypt_key)
        self._encrypted = True
        
    def __repr__(self):
        if self._encrypted:
            return("You can't read the content")
        else:
            return self._arr.__repr__()