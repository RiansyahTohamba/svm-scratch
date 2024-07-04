import numpy as np
arr = np.arange(8)
newarr = np.where(arr%2==0, arr*10,arr)
print(newarr)