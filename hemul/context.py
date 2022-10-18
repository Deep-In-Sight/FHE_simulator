from .cipher import *
from .scheme import *

def set_all(logp, logq, logn):
    myring = Ring(seed=1234)
    parms = Parameters(logp = logp, logq = logq, logn = logn)
    nslots = 2**parms.logn
    context = Context(parms, myring)

    sk = context.generate_secret_key()
    keys = {"mult":context.generate_mult_key(),
        "rot":{'1':'hi1',
               '2':'hi2',
               '4':'hi4',
               '8':'hi8'}}
    ev = Evaluator(keys, context) # Evaluator도 그냥 context만 넣게 할까? 

    encoder = Encoder(context)
    encryptor = Encryptor(context)
    decryptor = Decryptor(sk)

    return context, ev, encoder, encryptor, decryptor
