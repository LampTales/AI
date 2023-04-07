import time

import numpy as np

a = np.zeros((8, 8))
b = np.array(a)
b[1][2] = 5
print(a)
print(b)