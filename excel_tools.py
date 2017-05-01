"""
Excel Tools Used as Excel Subs and UDFs
"""

import xlwings as xw
import numpy as np
from typing import *
import products
from inspect import signature, BoundArguments

ProductManager, ModelPoint = products.ProductManager, products.ModelPoint


def make_kwargs(keywords: List[str], values: List, f: Callable=None) -> dict:
    keywords: List[str] = [x.lower() for x in keywords]
    d = dict(zip(keywords, values))
    if f is not None:
        sig = signature(f)
        params = sig.parameters
        bind_args: BoundArguments = sig.bind_partial(**d)
        args = bind_args.arguments
        for k, v in args.items():
            if isinstance(v, str):
                annotation = params[k].annotation
                if annotation is not str:
                    args[k] = annotation.from_str(v)
        return args
    else:
        return d


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('y', np.array, ndim=2)
@xw.ret(ndim=2)
def pSum(x, y=0):
    """ add 2 vertical arrays and return a vertical array """
    return x.sum(axis=1).reshape((x.shape[0], 1)) + y


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('y', np.array, ndim=2)
@xw.ret(ndim=2)
def pProd(x, y=1):
    """ multiply 2 vertical arrays and return a vertical array """
    return x.prod(axis=1).reshape((x.shape[0], 1)) * y


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('bottomUp', doc="if TRUE then cumsum from bottom default True")
@xw.ret(ndim=2)
def pCumsum(x, bottomUp=True):
    """ cumsum a vertical array """
    return x[::-1, :].cumsum(axis=0)[::-1, :] if bottomUp else x.cumsum(axis=0)


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('bottomUp', doc="if TRUE then cumprod from bottom default True")
@xw.ret(ndim=2)
def pCumprod(x, bottomUp=False):
    """ cumprod a vertical array """
    return x[::-1, :].cumprod(axis=0)[::-1, :] if bottomUp else x.cumprod(axis=0)


def modelPoint(keywords, values) -> ModelPoint:
    keywords: List[str] = [x.lower() for x in keywords]
    # k = ['sex', 'age']
    # v = [0, 10]
    # d = {"sex": 0, 'age': 10}
    # ModelPoint(sex=0, age=10)
    return ModelPoint(**dict(zip(keywords, values)))


@xw.func
@xw.arg('prodId', doc="the id of the Product")
@xw.arg('methodName', doc="the method name of the py function")
@xw.arg('mp_keywords', doc="ModelPoint init keywords")
@xw.arg('mp_values', doc="ModelPoint init values")
@xw.arg('keywords', np.array)
@xw.arg('values', np.array)
@xw.arg('vertical', doc="shall we return a vertical array? default True")
@xw.ret(ndim=2)
def productArrayOf(prodId, methodName, mp_keywords, mp_values, keywords=None, values=None, vertical=True):
    global ProductManager
    mp = modelPoint(mp_keywords, mp_values)
    product = ProductManager.PRODUCTS[int(prodId)]
    f: Callable = getattr(product, methodName)
    kwarg = make_kwargs(keywords, values, f)
    x = f(**kwarg, mp=mp)
    return x if not vertical else x.reshape((x.shape[0], 1))


@xw.func
@xw.arg('prodId', doc="the id of the Product")
def productName(prodId)->str:
    return ProductManager.PRODUCTS[prodId].prod_name


if __name__ == "__main__":
    xw.serve()
