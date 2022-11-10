from ast import (Module, FunctionDef, arguments, arg, 
                Expr, Constant, Assign, Name, Store, 
                Call, Attribute, Load, BinOp, Sub, Mult, Div, Return)

Module(
  body=[
    FunctionDef(
      name='fhe_var',
      args=arguments(
        posonlyargs=[],
        args=[
          arg(arg='ctxt')],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Expr(
          value=Constant(value='Auto generated\n    ')),
        Assign(
          targets=[
            Name(id='m', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='algo', ctx=Load()),
              attr='fhe_mean',
              ctx=Load()),
            args=[
              Name(id='ctxt', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='diff', ctx=Store())],
          value=BinOp(
            left=Name(id='ctxt', ctx=Load()),
            op=Sub(),
            right=Name(id='m', ctx=Load()))),
        Assign(
          targets=[
            Name(id='result', ctx=Store())],
          value=BinOp(
            left=Call(
              func=Attribute(
                value=Name(id='algo', ctx=Load()),
                attr='fhe_sum_reduce',
                ctx=Load()),
              args=[
                BinOp(
                  left=Name(id='diff', ctx=Load()),
                  op=Mult(),
                  right=Name(id='diff', ctx=Load()))],
              keywords=[]),
            op=Div(),
            right=Call(
              func=Name(id='len', ctx=Load()),
              args=[
                Name(id='ctxt', ctx=Load())],
              keywords=[]))),
        Return(
          value=Name(id='result', ctx=Load()))],
      decorator_list=[])],
  type_ignores=[])