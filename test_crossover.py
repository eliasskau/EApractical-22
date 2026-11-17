import numpy as np
from GA import crossover

p1 = np.zeros(10, dtype=int)
p2 = np.ones(10, dtype=int)
print("parent 1:", p1)
print("parent 2:", p2)
print("child:   ", crossover(p1, p2))
print("parent 1 after:", p1)
