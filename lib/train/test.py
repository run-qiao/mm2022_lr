import random
import numpy as np
a = np.zeros(5)
for i in range(1000000):
	a[random.randint(0,4)] += 1

print(a)
