import numpy as np
from GA import mutate

for t in range(5):
    x = np.zeros(50, dtype=int)
    x = mutate(x)
    print("number of flipped bits:", sum(x))