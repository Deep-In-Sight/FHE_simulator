import numpy as np
from hemul.ciphertext import Ciphertext, Plaintext, CiphertextStat
from hemul.utils import check_compatible, key_hash
from hemul.cipher import Context


class Encoder():
    def __init__(self, context):
        self.logp = context.params.logp
        self.nslots = context.params.nslots

    def encode(self, arr, logp=None, nslots=None):
        if logp:
            self.logp = logp
        if nslots:
            self.nstlos = nslots

        assert(self.logp != None), 'Ptxt scale not set'

        return Plaintext(arr=arr, logp = self.logp, nslots = self.nslots)


class Encryptor():
    def __init__(self, context:Context):
        self._context = context
        self._enc_key = self._context.enc_key
        self.enckey_hash = key_hash(self._enc_key)
    
    def encrypt(self, arr):
        # How to determine if I want Ciphertext or CiphertextStat?
        ctxt = CiphertextStat(self._context.params.logp, 
                              self._context.params.logq,
                              self._context.params.logn)
        
        #encoded = _stringify(arr)
        ctxt._set_arr(self.enckey_hash, arr)
        # TODO: Need to determine nslots 
        return ctxt

class Decryptor():
    def __init__(self, secret_key):
        self._secret_key = secret_key
        self._sk_hash = key_hash(secret_key)
    
    def decrypt(self, ctxt):
        if ctxt._enckey_hash == self._sk_hash and ctxt._encrypted:
            return ctxt._arr
        else:
            raise ValueError("You have a wrong secret key")



class FHE_OP():
    def _change_mod(self, ctxt:Ciphertext, logq):
        """
        proxy for Scheme.change_mod
        """
        ctxt.logq = logq
        self._counter.mod_switch(ctxt)

    @staticmethod
    @check_compatible
    def _add(ctxt1:Ciphertext, ctxt2:Ciphertext):
        """proxy for HEAAN.ring.add() and ring.addAndEqual()
        """
        return ctxt1._arr + ctxt2._arr

    @staticmethod
    @check_compatible
    def _add_plain(ctxt:Ciphertext, ptxt:Plaintext):
        """proxy for HEAAN.ring.addConst() and ring.addConstAndEqual()
        """
        return ctxt._arr + ptxt._arr

    @staticmethod
    @check_compatible
    def _sub(ctxt1:Ciphertext, ctxt2:Ciphertext):
        """proxy for HEAAN.ring.sub() and ring.subAndEqual1,2()
        """
        return ctxt1._arr - ctxt2._arr   

    @staticmethod
    @check_compatible
    def _sub_plain(ctxt:Ciphertext, ptxt:Plaintext):
        """proxy for HEAAN.ring.sub() and ring.subAndEqual1,2()
        """
        return ctxt._arr - ptxt._arr

    def _mult(self, ctxt1:Ciphertext, ctxt2:Ciphertext):
        """
        """
        if not ctxt1._ntt:
            self.switch_ntt(ctxt1)
        if not ctxt2._ntt:
            self.switch_ntt(ctxt2)        

        self._counter.multc(ctxt1)
        return ctxt1._arr * ctxt2._arr

    def _mult_by_plain(self, ctxt, ptxt):
        self._counter.multp(ctxt)
        return ctxt._arr * ptxt._arr

    def _leftrot(self, ctxt:Ciphertext, r:int):
        """
        """
        self._counter.rot(ctxt)
        return np.roll(ctxt._arr, -r)