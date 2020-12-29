# cod-otoliths

Predicting age of cod otoliths.

In folders- 2014  2015  2016  2017  2018  Extra - there are 12311 .JPG files. 12311/6 = 2051 unique images, and 1984 unique directories.

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species              | Predict    |testLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f | classWeights |
| ---------------------| -----------|--------|------|------|-----|-----|---------|---------|--------------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124 |0.262|x    |8875     | linear  | x | 
| Salmon               | sea age    | -"-    |0.239 |0.141 |0.822|x    |9073     | linear  | x |
| Salmon B4            | river age  |0.359   |0.359 |19.58 |0.618|x    |6246     | linear  | x |
| ~~Cod B4~~                | age        |0.0297  |0.0297|1.588 |0.984|x    |6330     | linear | x |
| Cod B4               | age        |0.8796  |0.8796|9.228 |0.52597|x    |1029     | linear | x |
| Cod B4               | age        |-  |-|- |-|x    |1985     | linear | x |


```
>>> unique, counts = np.unique(age, return_counts=True)
>>> len(age)
6330
>>> print("dataset ocurrence of each class:"+str(dict(zip(unique, counts))))
test ocurrence of each class:
{0: 12, 1: 222, 2: 300, 3: 432, 4: 534, 5: 576, 6: 804, 7: 936, 8: 732, 9: 570, 10: 324, 11: 282, 12: 366, 13: 102, 14: 96, 15: 24, 16: 18}
>>>
>>> unique, counts = np.unique(train_age, return_counts=True)
>>> dict(zip(unique, counts))
{0: 9, 1: 151, 2: 209, 3: 311, 4: 365, 5: 404, 6: 565, 7: 639, 8: 515, 9: 398, 10: 241, 11: 210, 12: 253, 13: 68, 14: 60, 15: 20, 16: 13}
>>> unique, counts = np.unique(val_age, return_counts=True)
>>> dict(zip(unique, counts))
{0: 3, 1: 35, 2: 43, 3: 57, 4: 79, 5: 88, 6: 132, 7: 135, 8: 104, 9: 86, 10: 41, 11: 43, 12: 60, 13: 16, 14: 24, 15: 2, 16: 2}
>>> unique, counts = np.unique(test_age, return_counts=True)
>>> dict(zip(unique, counts))
{1: 36, 2: 48, 3: 64, 4: 90, 5: 84, 6: 107, 7: 162, 8: 113, 9: 86, 10: 42, 11: 29, 12: 53, 13: 18, 14: 12, 15: 2, 16: 3}
```
