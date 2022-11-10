import numpy as np
from hemul.stats import Statistics
from hemul.ciphertext import CiphertextStat
from time import sleep

class Column(CiphertextStat):
    def __init__(self, ctxt, host):
        self.logp = ctxt.logp
        self.logq = ctxt.logq
        self.logn = ctxt.logn
        self._arr = ctxt._arr
        self._encrypted = ctxt._encrypted
        self._enckey_hash = ctxt._enckey_hash
        self._n_elements = ctxt._n_elements
        self._basic_stats = ctxt._basic_stats
        self.host = host
        sleep(0.1)

    def var(self):
        _var = self.host._stats.var(self)
        self._basic_stats['var'] = _var
        sleep(2.4)
        return _var

class CtxtFrame():
    def __init__(self, agents=None, enc_key=None, mul_key=None):
        if agents is not None:
            self._set_agetns(agents)

        self._columns = {}

    def _set_agetns(self, agents):
        self.ev = agents['evaluator']
        self.encoder = agents['encoder']
        self.encryptor = agents['encryptor']
        self.nslots = self.ev.context.params.nslots
        #self.n_elements = self.ev.params.n_elements
        
        self._init_stats()

    def _init_stats(self):
        self._stats = Statistics(self.ev, self.encoder)

    def add_column(self, name, ctxt):
        # Gaussian error
        # 적절한 크기 체크 필요 
        ctxt._arr += np.random.normal(0, ctxt._arr.max()*1e-9, size=ctxt._arr.shape)
        self._columns.update({name: self._gen_column(ctxt)})

    def _gen_column(self, ctxt):
        return Column(ctxt, self)

    def __len__(self):
        return self.n_elements

    #def to_df(self, decryptor):
    # """Decrypt and export dataset to pandas dataframe"""
    #
    # def to_numpy(self, decryptor):
    #"""Decrypt and export dataset to numpy array"""
    #"""

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        elif isinstance(key, int):
            return self._columns[key]
        else:
            raise TypeError("key must be either str or int")
        
