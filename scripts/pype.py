import ast, inspect 
from modifier import modifiers
import static_analyzer
from hemul.ciphertext import Ciphertext
import hemul.context as hcontext
import types
from hemul.algorithms import Algorithms
from time import sleep

def jit(func, dump=True, verbose=False):
    """Parse func's source code and modify it to FHE-friendly code.

        (optionally) dump new code in a func_name.py file.
    """
    # get source code
    source_code = inspect.getsource(func)
    source_code = '\n'.join(source_code.splitlines()[1:]) # remove decorator
    # parse source code
    tree = ast.parse(source_code)
    # infer data type
    tree = static_analyzer.infer_dtype(source_code, tree)
    # apply all modifiers (NumpyReplacer, BinOpReplacer, AutoRescaler, and so on..)
    for visitor in modifiers:
        visitor().visit(tree)
        ast.fix_missing_locations(tree)

    tree = ast.fix_missing_locations(tree)
    code_object = compile(tree, filename="", mode="exec")
    new_f = types.FunctionType(code_object.co_consts[1], func.__globals__)
    
    if dump:
        with open(func.__name__+"_md.py", "w") as f:
            f.write(ast.unparse(tree))
    
    # return compiled object of the modified source code
    if verbose:
        print(ast.dump(tree, indent=2))
    return new_f

def jit_verbose(func, dump=True):
    """Parse func's source code and modify it to FHE-friendly code.

        (optionally) dump new code in a func_name.py file.
    """
    # get source code
    source_code = inspect.getsource(func)
    source_code = '\n'.join(source_code.splitlines()[1:]) # remove decorator
    # parse source code
    tree = ast.parse(source_code)
    # infer data type
    tree = static_analyzer.infer_dtype(source_code, tree)
    # apply all modifiers (NumpyReplacer, BinOpReplacer, AutoRescaler, and so on..)
    for visitor in modifiers:
        visitor().visit(tree)
        ast.fix_missing_locations(tree)

    tree = ast.fix_missing_locations(tree)
    code_object = compile(tree, filename="", mode="exec")
    new_f = types.FunctionType(code_object.co_consts[1], func.__globals__)
    
    if dump:
        with open(func.__name__+"_md.py", "w") as f:
            f.write(ast.unparse(tree))
    
    # return compiled object of the modified source code
    print(ast.dump(tree, indent=2))
    return new_f

def set_all(logp, logq, logn):
    #global ev, encoder, encryptor, algo
    (context, ev, encoder, encryptor, decryptor) = hcontext.set_all(logp, logq, logn)
    algo = Algorithms(ev, encoder)
    return ev, algo, encoder, encryptor, decryptor