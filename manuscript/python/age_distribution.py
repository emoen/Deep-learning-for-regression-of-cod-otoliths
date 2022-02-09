len(age)
1985
>>> unique, counts = np.unique(age, return_counts=True)
>>> age_range_freq = dict(zip(unique, counts))
test ocurrence of each class:
{0: 7, 1: 149, 2: 187, 3: 209, 4: 204, 5: 298, 6: 217, 7: 228, 8: 162, 9: 120, 10: 98, 11: 52, 12: 33, 13: 13, 14: 4, 15: 2, 16: 2

import matplotlib.pyplot as plt
import numpy as np

N_points = len(age)
n_bins = 2
x=unique
y=counts
plt.bar(x, height=y)
plt.show()
