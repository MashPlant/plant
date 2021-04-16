from . import plant
from typing import *
import os.path
import numpy as np

def _wrap(args: List[np.ndarray]):
  ret = [0] * 3 * len(args)
  for i, x in enumerate(args):
    ret[i * 3] = x.ctypes.data
  return ret

def parallel_init(th: int = 0):
  plant.parallel_init(th)

class Func:
  def __init__(self, path: str, name: Optional[str] = None):
    self.base = plant.FuncBase(path, name if name is not None else os.path.splitext(os.path.basename(path))[0])

  def __call__(self, *args: np.ndarray):
    self.base(_wrap(args))

  def eval(self, args: List[np.ndarray], n_discard: int = 1, n_repeat: int = 3, timeout: int = 1000) -> Tuple[float, bool]:
    return self.base.eval(_wrap(args), n_discard, n_repeat, timeout)
