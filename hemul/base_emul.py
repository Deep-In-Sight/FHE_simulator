import numpy as np
from hemul.ciphertext import Ciphertext, Plaintext
from hemul.utils import check_compatible

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