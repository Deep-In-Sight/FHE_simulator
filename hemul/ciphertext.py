from logging.config import valid_ident
import numpy as np
from .errors import InvalidParamError
from .cipher import Parameters

class CipherABC():
    def __init__(self, logp:int=None, logn:int=None, nslots:int=None):
        if logp:
            self.logp = logp
        if logn and nslots:
            assert 2**logn == nslots
        if logn:
            self.logn = logn 
        if nslots:
            self.nslots = nslots 

    @property
    def logn(self):
        """automatically update on changing logn"""
        return self._logn

    @logn.setter
    def logn(self, val):
        self._logn = val
        self._nslots = 2**self._logn

    @property
    def nslots(self):
        return self._nslots

    @nslots.setter
    def nslots(self, val):
        self._nslots = val
        # TODO: need to deal with inexact nslots
        self._logn = int(np.log2(self._nslots))

    def __len__(self):
        return self.nslots

class Plaintext(CipherABC):
    """
    Plaintext의 logp는 미리 정해질 필요 없음. 
    ptxt는 대부분 mult_p 혹은 add_p를 위한 임시 변수이므로, 
    계산에 참여하는 ctxt의 ctxt.logp를 받으면 충분함.
    """
    def __init__(self, arr=None, logp=None, logn=None, nslots=None):
        super().__init__(logp=logp, logn=logn, nslots=nslots)
        assert logn or nslots, "Please specify logn or nslots"

        self._encrypted=None
        if arr is not None:
            self._set_arr(arr)
    
    def _set_arr(self, arr, n_elements=None):
        assert len(arr) <= self.nslots, "array longer than Nslots"

        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if not np.issubdtype(arr.dtype, np.number):
            print("Need a numeric type")
            raise ValueError
        else:
            self._arr = np.zeros(self.nslots)
            self._arr[:len(arr)] = arr
        if n_elements is not None:
            self._n_elements = n_elements
        else:
            self._n_elements = len(arr)

        self._encrypted = False
        

class Ciphertext(CipherABC):
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
        super().__init__()
        self.logp = None
        self.logq = None
        self._logn = None
        self._nslots = None
        self._ntt = False
        self.level = 0
        
        if len(args) == 1:
            if isinstance(args[0], Ciphertext):
                self.__init_with_ctxt__(args[0])
            elif isinstance(args[0], Parameters):
                self.__init_with_parmeters(args[0])
        elif len(args) == 3:
            self.__init_with_tuple(*args)
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
            self._verify_params()

        if 'arr' in kwargs.keys():
            self._set_arr(kwargs['arr'])
        else:
            self._arr = np.zeros(self.nslots)
        
    def __init_with_ctxt__(self, ctxt):
        """create a new ciphertext with the same properties as ctxt"""
        self.logp = ctxt.logp
        self.logq = ctxt.logq
        self.logn = ctxt.logn
        #print(ctxt.logp, ctxt.logq, ctxt.logn)
        
    def __init_with_parmeters(self, parms):
        self.logp = parms['logp']
        self.logq = parms['logq']
        self.logn = parms['logn']
        
    def __init_with_tuple(self, *arg):
        self.logp, self.logq, self.logn = arg
        
    def _verify_params(self):
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
        self._encrypted = True
        self._valid_slots = None
    
    def _set_arr(self, enckey_hash, arr, n_elements=None):
        assert len(arr) <= self.nslots, "array longer than Nslots"

        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if not np.issubdtype(arr.dtype, np.number):
            print("Need a numeric type")
            raise ValueError
        else:
            self._arr = np.zeros(self.nslots)
            self._arr[:len(arr)] = arr
        if n_elements is not None:
            self._n_elements = n_elements
        else:
            self._n_elements = len(arr)
        self._enckey_hash = enckey_hash
        self._encrypted = True
        
    def __repr__(self):
        if self._encrypted:
            return("You can't read the content")
        else:
            if self._valid_slots:
                return self._arr[self._valid_slots].__repr__()
            else:
                return self._arr.__repr__()
