import numbers
from .ciphertext import CiphertextStat, Plaintext
from .errors import ScaleMisMatchError, LengthMisMatchError
import numpy as np
"""
HEAAN's Ciphertext class has two construtors.
1: 	Ciphertext(long logp = 0, long logq = 0, long n = 0);
2:	Ciphertext(const Ciphertext& o);

"""

class Call_counter():
    """Count function call and store their parameters
        for later cost analysis.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._multp = []
        self._multc = []
        self._rot = []
        self._bootstrap = []
        self._mod_switch =[]
        self._rescale = []
        self._ntt_switch = []

    def get(self):
        return {"multp":self._multp,
                "multc":self._multc,
                "rot":self._rot,
                "bootstrap":self._bootstrap,
                "mod_switch":self._mod_switch,
                "rescale":self._rescale,
                "ntt_switch":self._ntt_switch}

    @staticmethod
    def _get_props(ctxt):
        return {"logq":ctxt.logq, 
                "logp":ctxt.logp,
                "logn":ctxt.logn,
                "ntt":int(ctxt._ntt)}
                
    def ntt_switch(self, ctxt):
        self._ntt_switch.append(self._get_props(ctxt))

    def rescale(self, ctxt):
        self._rescale.append(self._get_props(ctxt))

    def multp(self, ctxt):
        self._multp.append(self._get_props(ctxt))

    def multc(self, ctxt):
        self._multc.append(self._get_props(ctxt))

    def rot(self, ctxt):
        self._rot.append(self._get_props(ctxt))

    def bootstrap(self, ctxt):
        self._bootstrap.append(self._get_props(ctxt))

    def mod_switch(self, ctxt):
        self._mod_switch.append(self._get_props(ctxt))


def key_hash(key):
    return hash(str(key*np.conj(key)))

def secret_key_match_encryptor_key(enc, sk):
    """키 pair를 conjigate로 만들자! 
    """
    return key_hash(sk) == key_hash(enc)


### Decorator
def check_compatible(func):
    def func_wrapper(*args, **kwargs):
        """also applies to ptxt
        """
        ctxt1, ctxt2 = args
        goodp = ctxt1.logp == ctxt2.logp
        if isinstance(ctxt2, Plaintext):
            goodq = True
        else:
            goodq = ctxt1.logq == ctxt2.logq
        
        if goodp and goodq: 
            return func(*args, **kwargs)
        else:
            if not goodp: print(f"ctxt1.logp: {ctxt1.logp}, ctxt2.logp: {ctxt2.logp}")
            if not goodq: print(f"ctxt1.logq: {ctxt1.logq}, ctxt2.logq: {ctxt2.logq}")
            raise ScaleMisMatchError
    return func_wrapper

def check_plain_length(func):
    def func_wrapper(*args, **kwargs):
        ctxt, ptxt = args
        try:
            ll = len(ptxt)
            if ll == ctxt._n_elements:
                arr = np.zeros(ctxt.nslots)
                arr[:ll] = ptxt
                ptxt = arr
        except:
            if isinstance(ptxt, numbers.Number):
                pass
        else:
            raise LengthMisMatchError
        finally:
            return func(ctxt, ptxt, **kwargs)
        
    return func_wrapper


class Checker():
    """Error checker"""
    def __init__(self, atol=1e-6, rtol=1e-4):
        self.atol = atol
        self.rtol = rtol

    def close(self, v1, v2, atol=None, rtol=None):
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
            
        if np.isclose(v1,v2,atol=atol):
            print(f"Absolute error within: {atol:.1E}")
        else:
            print(f"Too large absolute error (> {atol:.1E})")
        if np.isclose(v1,v2,rtol=rtol):
            print(f"Relative error within: {rtol:.1E}")
        else:
            print(f"Too large relative error (> {rtol:.1E})")