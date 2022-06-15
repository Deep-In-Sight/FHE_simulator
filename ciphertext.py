import numpy as np
from cipher import key_hash

class Ciphertext():
    def __init__(self, logp, logq, nslots):
        """Base class for any ciphertext
        Currently following the CKKS convention -> should be changed.
        """
        self.logp = logp
        self.logq = logq
        self.nslots = nslots
        self.level = 0
        pass

class CiphertextStat(Ciphertext):
    def __init__(self, arr, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self._min = None  
        self._max = None
        self._mean = None
        self._set_arr(arr)
        self._n_elements = self._arr.__len__()
        self._encrypted = False
        self._enckey_hash = None
    
    def _set_arr(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if not np.issubdtype(arr.dtype, np.number):
            print("Need a numeric type")
            raise ValueError
        else:
            self._arr = arr

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