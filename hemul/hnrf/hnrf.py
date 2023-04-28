import numpy as np
import pickle
from typing import List#, Callable
#from .utils import extract_diagonals, pad_along_axis
from .nrf import NeuralDT, NeuralRF

from hemul.heaan import he

_all__ = ['to_list_and_duplicate', 'to_list_and_pad', 'HomomorphicModel', 'HEDT',
           'HNRF']

def to_list_and_duplicate(array):
    """Takes an array, append a 0 then copies itself once more.

    This step is necessary to do matrix multiplication with Galois rotations.
    This is used on the bias of the comparator.
    """
    array = list(array)
    array = array + [0] + array
    return array

def to_list_and_pad(array):
    """Takes an array, append len(array) of zeros 

    This step is necessary to do matrix multiplication with Galois rotations.
    This is used on the diagonal vectors.
    """
    array = list(array)
    array = array + [0] * (len(array) - 1)
    return array


class HomomorphicModel:
    """Base class for Homormorphic Decision Trees and Random Forest.

    As the Homomorphic Evaluator will only need weights, and comparator
    for the Homomorphic Featurizer, a model should only return these two.
    """
    def return_weights(self):
        return self.b0, self.w1, self.b1, self.w2, self.b2

    def return_comparator(self) -> np.ndarray:
        """Returns the comparator, which is a numpy array of the comparator,
        with -1 indices for null values. The array is repeated for Galois
        rotations before the multiplication."""

        comparator = list(self.comparator)
        comparator = comparator + [-1] + comparator
        comparator = np.array(comparator)
        return comparator

def gen_w1_dprecated(weight):
    weight = pad_along_axis(weight, weight.shape[0], axis=1)
    weight = extract_diagonals(weight)
    return [to_list_and_pad(weight[i]) for i in range(len(weight))]

def gen_w1(weight):
    """placeholder"""
    return None

class HEDT(HomomorphicModel):
    """Homomorphic Decision Tree, which extracts appropriate weights for
    homomorphic operations from a Neural Decision Tree."""

    def __init__(self, w0, b0, w1, b1, w2, b2):
        # We first get the comparator and set to -1 the rows that were padded
        comparator = w0
        padded_rows = (comparator.sum(axis=1) == 0)

        # We then get the indices of non padded rows
        comparator = comparator.argmax(axis=1)
        comparator[padded_rows] = -1
        self.comparator = comparator

        self.n_leaves = w1.shape[0]

        # We add a 0 then copy the initial
        self.b0 = to_list_and_duplicate(b0)

        # For weights, we first pad the columns, then extract the diagonals, and pad them
        self.w1 = gen_w1(w1)
        self.b1 = to_list_and_pad(b1)

        self.w2 = [to_list_and_pad(w2[c]) for c in range(len(w2))]

        self.b2 = [to_list_and_pad(([b2[c] / self.n_leaves]) * self.n_leaves) for c in range(len(b2))]

    @classmethod
    def from_neural_tree(cls, neural_tree: NeuralDT):
        return cls(neural_tree.return_weights())

class HNRF(HomomorphicModel):
    """Homomorphic Random Forest"""
    def __init__(self, neural_rf: NeuralRF, device="cpu"):

        homomorphic_trees = [HEDT(w0, b0, w1, b1, w2, b2)
                             for (w0, b0, w1, b1, w2, b2) in zip(*neural_rf.return_weights())]

        B0, W1, B1, W2, B2 = [], [], [], [], []
        comparator = []

        for h in homomorphic_trees:
            b0, w1, b1, w2, b2 = h.return_weights()
            B0 += b0
            W1.append(w1)
            B1 += b1
            W2.append(w2)
            B2.append(b2)
            comparator += list(h.return_comparator())

        self.comparator = comparator

        W1 = list(np.concatenate(W1, axis=-1))
        W2 = list(np.concatenate(W2, axis=-1))
        B2 = list(np.concatenate(B2, axis=-1))

        # We will multiply each class vector with the corresponding weight for each tree
        weights = neural_rf.weights
        block_size = neural_rf.n_leaves_max * 2 - 1
        weights = [[weight.item()] * block_size for weight in weights]
        weights = np.concatenate(weights)

        W2 = [w2 * weights for w2 in W2]
        B2 = [b2 * weights for b2 in B2]

        self.b0 = B0
        self.w1 = W1
        self.b1 = B1
        self.w2 = W2
        self.b2 = B2

class HETreeFeaturizer:
    """Featurizer used by the client to encode and encrypt data.
       모든 Context 정보를 다 필요로 함. 
    """
    def __init__(self, comparator: np.ndarray,
                 scheme, 
                 ckks_parms,
                 use_symmetric_key=False):
        self.comparator = comparator
        self.scheme = scheme
        self._parms = ckks_parms
        self.use_symmetric_key = use_symmetric_key
        
    def encrypt(self, x: np.ndarray):
        features = x[self.comparator]
        features[self.comparator == -1] = 0
        features = list(features)

        ctx = self._encrypt(features)
        return ctx

    def _encrypt(self, val, n=None, logp=None, logq=None):
        if n == None: n = self._parms.n
        if logp == None: logp = self._parms.logp
        if logq == None: logq = self._parms.logq

        ctxt = he.Ciphertext()#logp, logq, n)
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self.scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
        del vv
        return ctxt

    def save(self, path:str):
        pickle.dump(self.comparator, open(path, "wb"))

class HETreeEvaluator:
    """Evaluator which will perform homomorphic computation"""

    def __init__(self, 
                 model,
                 #b0: np.ndarray, w1, b1, w2, b2,
                 scheme,
                 parms,
                 activation_coeffs: List[float], 
                 sk=None,
                 #polynomial_evaluator: Callable,
                 
                 #relin_keys: seal.RelinKeys, galois_keys: seal.GaloisKeys, scale: float,
                 do_reduction=True,
                 silent=False):
        """Initializes with the weights used during computation.

        Args:
            b0: bias of the comparison step

        """        
        b0, w1, b1, w2, b2 = model.return_weights()

        self.sk = sk
        self.scheme = scheme
        self.algo = he.SchemeAlgo(scheme)
        # scheme should hold all keys
        self.parms = parms
        
        self._activation_coeff = activation_coeffs
        self._activation_poly_degree = len(activation_coeffs) -1
        self.do_reduction = do_reduction

        # 10-degree activation -> up to 5 multiplications 
        logq_w1 = self.parms.logq - 5 * self.parms.logp
        logq_b1 = logq_w1 - self.parms.logp
        logq_b2 = logq_b1 - 5*self.parms.logp

        self.b0_ctx = self.encrypt(b0)
        #self.b0 = b0
        self.w1 = [self.to_double(w) for w in w1]
        #self.b1 = b1
        self.w2 = [self.to_double(w) for w in w2]
        self.b1_ctx = self.encrypt(b1, logq=logq_b1)
        self.b2_ctx = [self.encrypt(b, logq=logq_b2) for b in b2]

        if not silent: self.setup_summary()      
    
    def setup_summary(self):
        print("CKKS paramters:")
        print("---------------------------")
        print(f"n = {self.parms.n}")
        print(f"logp = {self.parms.logp}")
        print(f"logq = {self.parms.logq}")
        print(f"tanh activation polynomial coeffs = {self._activation_coeff}")
        print(f"tanh activation polynomial degree = {self._activation_poly_degree}")
        
        print("\nNeural RF")
        print("---------------------------")
        print(f"")
    
    def heaan_double(self, val):
        mvec = np.zeros(self.parms.n)
        mvec[:len(val)] = np.array(val)
        return he.Double(mvec)

    def decrypt_print(self, ctx, n=20):
        res1 = self.decrypt(ctx)
        print("_____________________")
        print(res1[:n])
        print(res1.min(), res1.max())
        print("---------------------")

    def decrypt(self, enc):
        temp = self.scheme.decrypt(self.sk, enc)
        arr = np.zeros(self.parms.n, dtype=np.complex128)
        temp.__getarr__(arr)
        return arr.real
        
    def encrypt_ravel(self, val, **kwargs):
        """encrypt a list
        """
        return self.encrypt(np.array(val).ravel(), **kwargs)

    def encrypt(self, val, n=None, logp=None, logq=None):
        if n == None: n = self.parms.n
        if logp == None: logp = self.parms.logp
        if logq == None: logq = self.parms.logq
            
        ctxt = he.Ciphertext()
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        self.scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
        del vv
        return ctxt
    
    def to_double(self, val):
        n = self.parms.n
        vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
        vv[:len(val)] = val
        return he.Double(vv)
        
    def activation(self, ctx):
        output = he.Ciphertext()
        output = he.Ciphertext()
        self.algo.function_poly(output, 
                    ctx, 
                    he.Double(self._activation_coeff), 
                    self.parms.logp, 
                    self._activation_poly_degree)
        return output        
        
    def __call__(self, ctx):
        # First we add the first bias to do the comparisons
        ctx = self.compare(ctx) # sub and activate 
        ctx = self.match(ctx)
        outputs = self.decide(ctx)
        if self.do_reduction:
            outputs = self.reduce(outputs)

        return outputs

    def compare(self, ctx, debug=False):
        """Calculate first layer of the HNRF
        
        ctx = featurizer.encrypt(x)
        
        Assuming n, logp, logq are globally available
        
        """
        b0_ctx = self.b0_ctx
        self.scheme.addAndEqual(ctx, b0_ctx)
        # Activation
        output = self.activation(ctx)
            
        del b0_ctx, ctx

        return output
    
    def _mat_mult(self, diagonals, ctx):
        """
        Take plain vector 
        """
        scheme = self.scheme
        n = self.parms.n
        logp = self.parms.logp
        #logq = self.parms.logq

        ctx_copy = he.Ciphertext()
        ctx_copy.copy(ctx)
        
        for i, diagonal in enumerate(diagonals):
            if i > 0: scheme.leftRotateFastAndEqual(ctx_copy, 1) # r = 1

            # Multiply with diagonal
            dd = he.Ciphertext()
            # Reduce the scale of diagonal
            scheme.multByConstVec(dd, ctx_copy, diagonal, logp)
            scheme.reScaleByAndEqual(dd, logp)
            
            if i == 0:
                mvec = np.zeros(n)
                temp = he.Ciphertext()
                scheme.encrypt(temp, he.Double(mvec), n, logp, ctx_copy.logq - logp)
            
            # match scale 
            scheme.addAndEqual(temp, dd)

            #print("temp",i)
            #self.decrypt_print(temp,10)
            
            del dd
        del ctx_copy
        return temp

    def match(self, ctx):
        """matrix multiplication, then activation.
        """
        output = self._mat_mult(self.w1, ctx)

        #print(f"MATCH:: 'output.logq', {output.logq} == {self.b1_ctx.logq}?")
        self.scheme.addAndEqual(output, self.b1_ctx)
        
        output = self.activation(output)
        return output

    def decide(self, ctx):
        """Applies the decisions homomorphically.

        For each class, multiply the ciphertext with the corresponding weight of that class and
        add the bias afterwards.
        """
        # ww와 bb도 미리 modDowntoAndEqual 가능 
        outputs = []

        for ww, bb in zip(self.w2, self.b2_ctx):
            output = he.Ciphertext()
            # Multiply weights
            self.scheme.multByConstVec(output, ctx, ww, ctx.logp)
            self.scheme.reScaleByAndEqual(output, ctx.logp)
            
            # Add bias
            self.scheme.addAndEqual(output, bb)
            
            outputs.append(output)
        return outputs

    def _sum_reduce(self, ctx, logn, scheme):
        """
        return sum of a Ciphertext (repeated nslot times)
        
        example
        -------
        sum_reduct([1,2,3,4,5])
        >> [15,15,15,15,15]
        """
        output = he.Ciphertext()
        
        for i in range(logn):
            
            if i == 0:
                temp = he.Ciphertext(ctx.logp, ctx.logq, ctx.n)
                
                scheme.leftRotateFast(temp, ctx, 2**i)
                scheme.add(output, ctx, temp)
            else:
                scheme.leftRotateFast(temp, output, 2**i)
                scheme.addAndEqual(output, temp)
        return output

    def reduce(self, outputs):
        logp = self.parms.logp
        scheme = self.scheme

        for i, output in enumerate(outputs):
            output = self._sum_reduce(output, self.parms.logn, self.scheme)

            mask = np.zeros(self.parms.n)
            mask[0] = 1
            mask_hedb = he.ComplexDouble(mask)
            if i == 0:
                scores = he.Ciphertext()
                scheme.multByConstVec(scores, output, mask_hedb, logp)
                scheme.reScaleByAndEqual(scores, logp)
            else:
                temp = he.Ciphertext()
                scheme.multByConstVec(temp, output, mask_hedb, logp)
                scheme.reScaleByAndEqual(temp, logp)
                scheme.rightRotateFastAndEqual(temp, i)
                scheme.addAndEqual(scores, temp)

        return scores

    @classmethod
    def from_model(cls, model,
                   scheme,
                   parms,
                   activation_coeffs: List[float],
                   do_reduction=False,
                   sk=None):
        """Creates an Homomorphic Tree Evaluator from a model, i.e a neural tree or
        a neural random forest. """
        b0, w1, b1, w2, b2 = model.return_weights()

        return cls(b0, w1, b1, w2, b2, scheme, parms, activation_coeffs, do_reduction, sk=sk)