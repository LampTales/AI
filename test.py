import time
import seaborn as sns; sns.set()
import numpy as np

a = []
b = []
edge = (3, 4)
a.append(edge)
b.append(edge)
edge[1] += 10
print(a)
print(b)
a[1][1] += 6
print(a)
print(b)