import sys, os
from os import path
import scipy as sp
import ctypes
from ctypes import POINTER, c_double, c_int, c_char_p, CDLL, byref
from ctypes.util import find_library
from subprocess import check_call

if sys.version_info[0] < 3:
    range = xrange
    from itertools import izip as zip
    _to_cstr = lambda s: s.encode("utf-8") if isinstance(s, unicode) else str(s)
else :
    _to_cstr = lambda s: bytes(s, "utf-8")
    basestring = (str, buffer)

dirname = path.dirname(path.abspath(__file__))
try :
    _clib = CDLL(path.join(dirname, 'src/_tclib.so'))
except:
    check_call(['make','-C', path.join(dirname, 'src')])
    _clib = CDLL(path.join(dirname, 'src/_tclib.so'))

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

fillprototype(_clib.tc_write, None, [POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int, c_char_p])
fillprototype(_clib.tc_read_size, None, [c_char_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)])
fillprototype(_clib.tc_read_content, None, [c_char_p, c_int, c_int, c_int, POINTER(c_double), POINTER(c_double)])


def tc_write(W, H, topk, output_name):
    assert W.dtype == sp.float64 and H.dtype == sp.float64
    assert W.shape[1] == H.shape[1]
    assert W.flags['C_CONTIGUOUS'] == True and H.flags['C_CONTIGUOUS'] == True
    m = W.shape[0]
    n = H.shape[0]
    k = W.shape[1]
    Wptr = W.ctypes.data_as(POINTER(c_double))
    Hptr = H.ctypes.data_as(POINTER(c_double))
    _clib.tc_write(Wptr, Hptr, c_int(m), c_int(n), c_int(k), topk, _to_cstr(output_name))

def tc_read(filename):
    m, n, k = c_int(0), c_int(0), c_int(0)
    _clib.tc_read_size(_to_cstr(filename), byref(m), byref(n), byref(k))
    W = sp.zeros((m.value, k.value), dtype=sp.float64)
    H = sp.zeros((k.value, n.value), dtype=sp.float64)
    Wptr = W.ctypes.data_as(POINTER(c_double))
    Hptr = H.ctypes.data_as(POINTER(c_double))
    _clib.tc_read_content(_to_cstr(filename), m, n, k, Wptr, Hptr)
    return W, sp.ascontiguousarray(H.T)


