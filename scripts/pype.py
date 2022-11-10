import ast, inspect 
from modifier import modifiers
import static_analyzer

def jit(func, dump=True):
    """Parse func's source code and modify it to FHE-friendly code.

        (optionally) dump new code in a func_name.py file.
    """
    # get source code
    source_code = inspect.getsource(func)
    # parse source code
    tree = ast.parse(source_code)
    # infer data type
    tree = static_analyzer.infer_dtype(source_code, tree)
    # apply all modifiers (NumpyReplacer, BinOpReplacer, AutoRescaler, and so on..)
    for visitor in modifiers:
        visitor().visit(tree)

    if dump:
        with open(func.__name__+"_md.py", "w") as f:
            f.write(ast.unparse(tree))
    
    # return compiled object of the modified source code
    return compile(tree, filename="<ast>", mode="exec")