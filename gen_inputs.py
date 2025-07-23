import numpy as np

a = np.random.rand(64, 16384).astype(np.int8)
b = np.random.rand(53248, 16384).astype(np.int8)
a_scales = np.random.rand(64, 512).astype(np.int8)
b_scales = np.random.rand(53248, 512).astype(np.int8)

np.save("tmp/a.npy", a)
np.save("tmp/a_scales.npy", a_scales)
np.save("tmp/b.npy", b)
np.save("tmp/b_scales.npy", b_scales)
