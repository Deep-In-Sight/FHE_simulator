class OfheContext():
    def __init__(self, **kwargs):
        
        #self.scheme = Ofhe...object
        pass
    
    def Ciphertext(self, value=None):
        if value is None:
            return self.scheme.CKKSSCiphertext()
        else:
            pass 
    
    def encrypt(self, value):
        # value에 대한 전처리
        ctxt = self.scheme.encrypt(value)
        return ctxt
    
    def encode(self, value):
        # Plaintext 만들기 
        
        
    def copy(self, ctxt):
        
        #new_ctxt = self.scheme.CKKSSCiphertext()
        #new_ctxt. value = ctxt.value
        
        return new_ctxt
    
    def add(self, ctxt1, ctxt2, inplace=False):
        # ctxt2가 Ciphertext이거나 np.array이거나 다 지원하면 편리
        if inplace:
            pass
        else:
            pass 
        return self.scheme.Evaladd(ctxt1, ctxt2)
    
    def lrot(self, ctxt, n):
        return self.scheme.Evalrot(ctxt, n)
    
    def mult(self, ctxt, val):
        """
        OFHE의 Polymorphism에 따라 암 것도 안 해도 될 수도. 
        """
        
    def sumslots(self, ctxt):
        pass
        # for  ..
            # self.lrot()
            # self.add()
        #return 
        
        