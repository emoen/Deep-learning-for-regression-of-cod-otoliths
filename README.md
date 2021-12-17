# cod-otoliths
In folders- 2014  2015  2016  2017  2018  Extra - there are 12311 .JPG files. 12311/6 = 2051 unique images (including files starting with ._), and 1984 unique directories.

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

### 5-fold training after adding 3000 images - testset 15%
| NN-config              | fold-1 (acc, mse) | fold-3 | fold-3  | fold-4 | fold-5 | mean MSE | mean ACC  | datset size |  |
| -----------------------| ------------------|--------|---------|--------|--------|----------|-----------|-------------|--|
| B4                     | [0.48629727959632874, 0.6488250494003296]|[0.4687076508998871, 0.6697127819061279]|[0.4820464551448822, 0.6631853580474854] |[0.4878818988800049, 0.6488250494003296]|[0.47346818447113037, 0.6579634547233582] |0.4216341924334377 |0.6971279373368147 |  5150       |  | 





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

## After addictional added 3000 images

![image](https://user-images.githubusercontent.com/1202040/140306617-8f266c26-2248-479e-a1a7-f60c6ccdf636.png)



<code>
{0: 11, 1: 380, 2: 509, 3: 504, 4: 617, 5: 795, 6: 532, 7: 544, 8: 476, 9: 319, 10: 214, 11: 120, 12: 54, 13: 26, 14: 7, 15: 4, 16: 1, 17: 1}
{0: 6, 1: 246, 2: 351, 3: 336, 4: 442, 5: 542, 6: 395, 7: 372, 8: 341, 9: 226, 10: 164, 11: 86, 12: 44, 13: 19, 14: 5, 15: 3, 16: 1}
</code>

