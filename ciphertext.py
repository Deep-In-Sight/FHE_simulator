import numpy as np
from errors import InvalidParamError
from cipher import Parameters

class Ciphertext():
    def __init__(self, *args, **kwargs):
        """Base class for any ciphertext
        Currently following the CKKS convention -> should be changed.
        
        Note
        ----
        Supports four 'constructors'
        
        1. Ciphertext(another_ctxt)
        2. Ciphertext(ctxt_params), where
            class Parameters():
                self.logp = 
                self.logq = 
                self.logn = 
                self. ...
                
        3. Ciphertext(30, 150, 12)
        4. Ciphertext(logp=30, logq=150, logn=12)
        """
        self.logn = None
        self.logp = None
        self.logq = None
        self.nslots = None
        self.level = 0
        
        if len(args) == 1:
            if isinstance(args[0], Ciphertext):
                self.__init_with_ctxt__(args[0])
            elif isinstance(args[0], Parameters):
                self.__init_with_parmeters(args[0])
        elif len(args) == 3:
            try:
                self.__init_with_tuple(*args)
            except:
                print("failed")
        else:
            try:
                self.logp = kwargs['logp']
                self.logq = kwargs['logq']
                self.logn = kwargs['logn']
            except NameError as err:
                print("Not valid set of kwargs are given. "
                      "try Ciphertext(logp, logq, logn)")
        
        if self.logp is not None or self.logq is not None or self.logn is not None:
            self._varify_params()
                
        
    def __init_with_ctxt__(self, ctxt):
        self.logp = ctxt.logp
        self.logq = ctxt.logq
        self.logn = ctxt.logn
        
    def __init_with_parmeters(self, parms):
        self.logp = parms['logp']
        self.logq = parms['logq']
        self.logn = parms['logn']
        
    def __init_with_tuple(self, *arg):
        self.logp, self.logq, self.logn = arg
        
    def _varify_params(self):
        """Todo"""
        if False:
            raise InvalidParamError
        

class CiphertextStat(Ciphertext):
    def __init__(self, *args, **kwargs):
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
        self._enckey_hash = None
    
    def _set_arr(self, enckey_hash, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if not np.issubdtype(arr.dtype, np.number):
            print("Need a numeric type")
            raise ValueError
        else:
            self._arr = arr
        
        self._n_elements = self._arr.__len__()
        self._enckey_hash = enckey_hash
        self._encrypted = True

        
    def __repr__(self):
        if self._encrypted:
            return("You can't read the content")
        else:
            return self._arr.__repr__()

