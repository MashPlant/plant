import plant
from plant import Array, Func
import numpy as np

M = N = K = 2048

a = np.random.uniform(size=(M, N)).astype(np.float32)
b = np.random.uniform(size=(N, K)).astype(np.float32)
c = np.dot(a, b)

a_gpu = Array.from_np(a).to_gpu()
b_gpu = Array.from_np(b).to_gpu()
c_gpu = Array.alloc(c.shape, ty=plant.F32, loc=plant.GPU)
f = Func('./matmul_gpu.so')
f(a_gpu, b_gpu, c_gpu)
c_cpu = c_gpu.to_cpu()

# test with numpy, relative error 1e-5
np.testing.assert_allclose(c, c_cpu.as_np(), rtol=1e-5)
# test with plant, absolute error 1e-2
c_cpu.assert_close(Array.from_np(c), threshold=1e-2)
