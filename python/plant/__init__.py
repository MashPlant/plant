from . import plant
from ctypes import POINTER, cast, c_int8, c_uint8, c_int16, c_uint16, c_int32, c_uint32, c_int64, c_uint64, c_float, c_double, c_void_p
from typing import List, Union, Optional
import os.path
import numpy as np

I8 = 0
U8 = 1
I16 = 2
U16 = 3
I32 = 4
U32 = 5
I64 = 6
U64 = 7
F32 = 8
F64 = 9
TY2STR = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']
STR2TY = {'int8': 0, 'uint8': 1, 'int16': 2, 'uint16': 3, 'int32': 4, 'uint32': 5, 'int64': 6, 'uint64': 6, 'float32': 8, 'float64': 9}
TY2C = [POINTER(c_int8), POINTER(c_uint8), POINTER(c_int16), POINTER(c_uint16), POINTER(c_int32), POINTER(c_uint32), POINTER(c_int64), POINTER(c_uint64), POINTER(c_float), POINTER(c_double)]

CPU = 0
GPU = 1

def parallel_init(th: int = 0):
  plant.parallel_init(th)

class Array:
  def __init__(self, impl: plant.Array):
    self.impl = impl
  @staticmethod
  def from_np(arr: np.ndarray) -> 'Array':
    return Array(plant.array_borrow(arr.shape, arr.ctypes.data, STR2TY[str(arr.dtype)]))
  @staticmethod
  def alloc(dims: List[int], ty: int, loc: int = CPU) -> 'Array':
    return Array(plant.array_alloc(dims, ty, loc))
  def as_np(self) -> np.ndarray:
    return np.ctypeslib.as_array(cast(c_void_p(self.ptr), TY2C[self.ty]), self.dims)
  def assert_close(self, rhs: 'Array', threshold: float = 0.0):
    return self.impl.assert_close(rhs.impl, threshold)
  def to_cpu(self) -> 'Array':
    return Array(self.impl.to_cpu())
  def to_gpu(self) -> 'Array':
    return Array(self.impl.to_gpu())
  @property
  def ptr(self) -> int:
    return self.impl.ptr
  @property
  def ty(self) -> int:
    return self.impl.ty
  @property
  def loc(self) -> int:
    return self.impl.loc
  @property
  def dims(self) -> List[int]:
    return self.impl.dims
  def __str__(self):
    return self.impl.__str__()

def _wrap(args: List[Union[np.ndarray, Array]]):
  ret = [0] * 3 * len(args)
  for i, x in enumerate(args):
    ret[i * 3] = x.ctypes.data if isinstance(x, np.ndarray) else x.ptr
  return ret

class Func:
  def __init__(self, path: str, name: Optional[str] = None):
    self.impl = plant.Func(path, name if name is not None else os.path.splitext(os.path.basename(path))[0])
  def __call__(self, *args: Union[np.ndarray, Array]):
    self.impl(_wrap(args))
  def eval(self, args: List[Union[np.ndarray, Array]], n_discard: int = 1, n_repeat: int = 3, timeout: int = 1000) -> (float, bool):
    return self.impl.eval(_wrap(args), n_discard, n_repeat, timeout)
