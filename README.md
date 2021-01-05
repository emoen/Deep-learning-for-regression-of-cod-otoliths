# cod-otoliths

Predicting age of cod otoliths.

In folders- 2014  2015  2016  2017  2018  Extra - there are 12311 .JPG files. 12311/6 = 2051 unique images (including files starting with ._), and 1984 unique directories.

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species              | Predict    |validLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f | classWeights |
| ---------------------| -----------|--------|------|------|-----|-----|---------|---------|--------------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124 |0.262|x    |8875     | linear  | x | 
| Salmon               | sea age    | -"-    |0.239 |0.141 |0.822|x    |9073     | linear  | x |
| Salmon B4            | river age  |0.359   |0.359 |19.58 |0.618|x    |6246     | linear  | x |
| ~~Cod B4~~                | age        |0.0297  |0.0297|1.588 |0.984|x    |6330     | linear | x |
| Cod B4               | age        |0.8796  |0.8796|9.228 |0.52597|x    |1029     | linear | x |
| Cod B4  (epoch41)    | age        |0.9695  |0.9695|- |0.5805|x    |1984     | linear | x |
| Cod B4  (epoch53)    | age        |0.9785  |0.9785|- |0.6174|x    |1984     | linear | x |
| Cod B4  (test-metric)| age        |0.7814  |0.7814|- |0.6409|x    |1984     | linear | x |


```
>>> len(age)
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
```

| ![cod-age_distribution.png](https://github.com/emoen/Deep-learning-for-cod-otoliths/blob/master/img/age_distribution.png) |
|:--:| 
| *Figure 1.:Age distribution of cod otoliths:* |
