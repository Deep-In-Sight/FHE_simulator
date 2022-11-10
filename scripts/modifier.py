import ast 
from ast import (Module, FunctionDef, arguments, arg, keyword,
                Expr, Constant, Assign, Name, Store, 
                Call, Attribute, Load, BinOp, Sub, Mult, Div, Return)

class RescaleAdder(ast.NodeTransformer):
    """Modifiers assume 'ev' and 'algo' are available in the namespace
    of execution."""
    def visit_Call(self, node):
        self._expr_statement = True
        self.generic_visit(node)
        self._expr_statement = False
        
        if isinstance(node.func, ast.Name):
            return node
        # ev.mult -> mult + rescale
        elif isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Name):
            if node.func.value.id in ["ev", "algo"]:
                if node.func.attr in ["mult", "mult_by_plain", "mean"]:
                  # node를 assign으로. 
                    return Call(
                              func=Attribute(
                                value=Name(id='ev', ctx=Load()),
                                attr='rescale_next',
                                ctx=Load()),
                              args=[
                                node],
                              keywords=[keyword(arg='inplace',
                                                value=Constant(value=False))])
                else:
                  return node
            else:
              return node
        else:
          return node

class ModMatcher(ast.NodeTransformer):
    def visit_Call(self, node):
        self._expr_statement = True
        self.generic_visit(node)
        self._expr_statement = False
        
        if isinstance(node.func, ast.Name):
            return node
        # ev.mult -> mult + rescale
        elif isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Name):
            if node.func.value.id in ["ev", "algo"]:
                if node.func.attr in ["mult", "add", "sub"]:
                  # node를 assign으로. 
                    return Call(
                              func=node.func,
                              args=[Call(
                                  func=Attribute(
                                    value=Name(id='ev', ctx=Load()),
                                    attr='match_mod',
                                    ctx=Load()),
                                  args=node.args,
                                  keywords=[keyword(arg='inplace',
                                                value=Constant(value=False))]
                                ), node.args[1]],
                                keywords=[keyword(arg='inplace',
                                                value=Constant(value=False))]
                                )
                else:
                  return node
            else:
              return node
        else:
          return node
  

def is_external_module_func(node):
    return isinstance(node.func, ast.Attribute) and \
              isinstance(node.func.value, ast.Name)

class NumpyReplacer(ast.NodeTransformer):
    def visit_Call(self, node):
        super().generic_visit(node)
        ## ordinary function


        if is_external_module_func(node):
            if node.func.value.id == "np":
                if node.func.attr == "mean":
                    return Call(
                        func=Attribute(
                          value=Name(id='algo', ctx=Load()),
                          attr="mean",
                          ctx=Load()),
                        args=node.args,
                        keywords=node.keywords)
                elif node.func.attr == "sum":
                    return Call(
                        func=Attribute(
                          value=Name(id='algo', ctx=Load()),
                          attr="sum_reduce",
                          ctx=Load()),
                          args=node.args,
                          keywords=node.keywords)
                else:
                    return node
            else:
                return node
        else:
          return node 

        # elif 
        #     print("Found a numpy call", node.func.value.id, node.func.attr)
        #     if node.func.attr == "sum":
        #         print(f'leaving {node.__class__.__name__}')
        #         return [Assign(
        #                     targets=[
        #                       Name(id='x', ctx=Store())],
        #                     value=Call(
        #                       func=Attribute(
        #                         value=Name(id='algo', ctx=Load()),
        #                         attr='fhe_sum_reduce',
        #                         ctx=Load()),
        #                       args=node.args,
        #                       keywords=node.keywords)), 
        #                     Call(
        #                       func=Attribute(
        #                         value=Name(id='ev', ctx=Load()),
        #                         attr='rescale_next',
        #                         ctx=Load()),
        #                       args=[Name(id='x', ctx=Load())],
        #                       keywords=['inplace=True'])] 
                 
        #     elif node.func.attr == "mean":
        #         print(f'leaving {node.__class__.__name__}')
        #         return Call(
        #                     func=Attribute(
        #                       value=Name(id='algo', ctx=Load()),
        #                       attr='fhe_mean',
        #                       ctx=Load()),
        #                     args=[
        #                       Name(id='data', ctx=Load())],
        #                     keywords=[])

class BinOpReplacer(ast.NodeTransformer):    
    def visit_BinOp(self, node):
        super().generic_visit(node)
        
        if isinstance(node.op, ast.Sub):
            #if is_op_cc(node):
            return Call(
                    func=Attribute(
                        value=Name(id='ev', ctx=Load()),
                        attr='sub',
                        ctx=Load()),
                    args=[
                        Name(id=node.left.id, ctx=node.left.ctx),
                        Name(id=node.right.id, ctx=node.right.ctx)],
                        keywords=[])
        elif isinstance(node.op, ast.Mult):
            #if is_op_cc(node):
            return Call(
                    func=Attribute(
                        value=Name(id='ev', ctx=Load()),
                        attr='mult',
                        ctx=Load()),
                    args=[
                        Name(id=node.left.id, ctx=node.left.ctx),
                        Name(id=node.right.id, ctx=node.right.ctx)],
                        keywords=[])
        elif isinstance(node.op, ast.Div):
            #if is_op_cc(node):
            return Call(
                    func=Attribute(
                        value=Name(id='ev', ctx=Load()),
                        attr='div_by_plain',
                        ctx=Load()),
                    args=[node.left,  
                         node.right,],
                        keywords=[])
        else:
            return node


# Order matters
modifiers = [BinOpReplacer, NumpyReplacer, RescaleAdder, ModMatcher]






# if op_cc(a,b):
#     # Match scale
#     if a.logp > b.logp:
#         add_rescale_a_b_before()
#     elif a.logp < b.logp:
#         add_rescale_b_a_before()
#     # Match level
#     if a.logp > b.logp:
#         add_modswitch_a_b_before()
#     elif a.logp < b.logp:
#         add_modswitch_b_a_before()

# def add_rescale_b_a_before():
#     pass

# def add_rescale_a_b_before():
#     pass

# def add_modswitch_b_a_before():
#     pass

# def add_modswitch_a_b_before():
#     pass
    
# from hemul.ciphertext import CipherText
# def op_cp(a,b):
#     return isinstance(a, Ciphertext) and if isinstance(b, Plaintext)
    
# def op_cc(a,b):
#     return isinstance(a, Ciphertext) and if isinstance(b, Ciphertext)