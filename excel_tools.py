"""
Excel Tools Used as Excel Subs and UDFs
"""

import xlwings as xw
import numpy as np
import actuarial_tools
from typing import *


__ExcelSupportedClasses__ = dict()


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('y', np.array, ndim=2)
@xw.ret(expand='table', ndim=2)
def vsum(x, y=0):
	""" add 2 vertical arrays and return a vertical array """
	return x.sum(axis=1).reshape((x.shape[0],1)) + y


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('y', np.array, ndim=2)
@xw.ret(expand='table', ndim=2)
def vprod(x, y=1):
	""" multiply 2 vertical arrays and return a vertical array """
	return x.prod(axis=1).reshape((x.shape[0],1)) * y


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('bottomUp', doc="if TRUE then cumsum from bottom default True")
@xw.ret(expand='table', ndim=2)
def vcumsum(x, bottomUp=True):
	""" cumsum a vertical array """
	return x[::-1,:].cumsum(axis=0)[::-1,:] if bottomUp else x.cumsum(axis=0)


@xw.func
@xw.arg('x', np.array, ndim=2)
@xw.arg('bottomUp', doc="if TRUE then cumprod from bottom default True")
@xw.ret(expand='table', ndim=2)
def vcumprod(x, bottomUp=True):
	""" cumprod a vertical array """
	return x[::-1,:].cumprod(axis=0)[::-1,:] if bottomUp else x.cumprod(axis=0)


class ExcelSupportManager:

	supportedClasses = __ExcelSupportedClasses__

	@classmethod
	def peakableMethod(cls, method):
		return method

	@classmethod
	def peakableClass(cls, clsObj):
		cls.supportedClasses[clsObj] = 12
		print(clsObj)
		print(__ExcelSupportedClasses__)
		return clsObj


@xw.func
def modelPoint(sex: int, age: int, policy_term: Union[str, int], payment_term: Union[str, int],
					policy_year: int=None, policy_month: int=None): 
	return actuarial_tools.ModelPoint(sex, age, policy_term, payment_term, policy_year, policy_month)

if __name__ == "__main__":
	a = 19
	print(ExcelSupportManager.supportedClasses)


