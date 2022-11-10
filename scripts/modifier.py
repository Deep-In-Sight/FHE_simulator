import ast 
from ast import (Module, FunctionDef, arguments, arg, 
                Expr, Constant, Assign, Name, Store, 
                Call, Attribute, Load, BinOp, Sub, Mult, Div, Return)

class RescaleAdder(ast.NodeTransformer):
    #self._expr_statement = False

    def visit_Call(self, node):
        self._expr_statement = True
        self.generic_visit(node)
        self._expr_statement = False
        
        if isinstance(node.func, ast.Name):
            return node
        # np -> algo
        elif isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Name):
            if node.func.value.id == "np":
                print("Found a numpy call", node.func.value.id, node.func.attr)
                if node.func.attr == "sum":
                    return Call(
                              func=Attribute(
                                value=Name(id='algo', ctx=Load()),
                                attr='fhe_sum_reduce',
                                ctx=Load()),
                              args=[
                                BinOp(
                                  left=Name(id='diff', ctx=Load()),
                                  op=Mult(),
                                  right=Name(id='diff', ctx=Load()))],
                              keywords=[])
                elif node.func.attr == "mean":
                    return Call(
                                func=Attribute(
                                  value=Name(id='algo', ctx=Load()),
                                  attr='fhe_mean',
                                  ctx=Load()),
                                args=[
                                  Name(id='data', ctx=Load())],
                                keywords=[])
        
        elif isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Name):
            if node.func.value.id == "ev":
                if node.func.value.attr in ["mult", "multp"]:
                    return Call(
                                func=Attribute(
                                  value=Name(id='ev', ctx=Load()),
                                  attr='rescale_next',
                                  ctx=Load()),
                                args=[],
                            keywords=[])


class NumpyReplacer(ast.NodeTransformer):
    def visit_Call(self, node):
        print(f'entering {node.__class__.__name__}')
        super().generic_visit(node)
        ## ordinary function
        if isinstance(node.func, ast.Name):#and \
            #node.func.value.id == 'np':
            print("Found a call", node.func.id, "skipping...")
            print(f'leaving {node.__class__.__name__}')
            return node
            #print(f"{node.func.value.id} . {node.func.attr}")
        elif isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Name):
            print("Found a numpy call", node.func.value.id, node.func.attr)
            if node.func.attr == "sum":
                print(f'leaving {node.__class__.__name__}')
                return Call(
                          func=Attribute(
                            value=Name(id='algo', ctx=Load()),
                            attr='fhe_sum_reduce',
                            ctx=Load()),
                          args=[
                            BinOp(
                              left=Name(id='diff', ctx=Load()),
                              op=Mult(),
                              right=Name(id='diff', ctx=Load()))],
                          keywords=[])
            elif node.func.attr == "mean":
                print(f'leaving {node.__class__.__name__}')
                return Call(
                            func=Attribute(
                              value=Name(id='algo', ctx=Load()),
                              attr='fhe_mean',
                              ctx=Load()),
                            args=[
                              Name(id='data', ctx=Load())],
                            keywords=[])

class BinOpReplacer(ast.NodeTransformer):    
    def visit_BinOp(self, node):
        super().generic_visit(node)
        
        if isinstance(node.op, ast.Sub):
            #if is_op_cc(node):
            print("This is sub")
            print(f'leaving {node.__class__.__name__}')
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
            print("This is mult")
            print(f'leaving {node.__class__.__name__}')
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
            print("This is div")
            print(f'leaving {node.__class__.__name__}')
            return Call(
                    func=Attribute(
                        value=Name(id='ev', ctx=Load()),
                        attr='mult_by_plain',
                        ctx=Load()),
                    args=[
                        Name(id=node.left.id, ctx=node.left.ctx),
                        BinOp(
                            left=Constant(value=1, kind=None),
                            op=Div(),
                            right=Name(id=node.right.id, ctx=node.right.ctx),
                         )     
                        ],
                        keywords=[])
        else:
            return node


modifiers = [RescaleAdder, NumpyReplacer]