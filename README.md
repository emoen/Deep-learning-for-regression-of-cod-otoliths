# cod-otoliths

Predicting age of cod otoliths

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species              | Predict    |testLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f | classWeights |
| ---------------------| -----------|--------|------|------|-----|-----|---------|---------|--------------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124 |0.262|x    |8875     | linear  | x | 
| Salmon               | sea age    | -"-    |0.239 |0.141 |0.822|x    |9073     | linear  | x |
| Salmon B4            | river age  |0.359   |0.359 |19.58 |0.618|x    |6246     | linear  | x |
| Cod B4               | age        |0.0297  |0.0297|1.588 |0.984|x    |6330     | linear | x |


```
>>> unique, counts = np.unique(age, return_counts=True)
>>> len(age)
6330
>>> print("test ocurrence of each class:"+str(dict(zip(unique, counts))))
test ocurrence of each class:{0: 12, 1: 222, 2: 300, 3: 432, 4: 534, 5: 576, 6: 804, 7: 936, 8: 732, 9: 570, 10: 324, 11: 282, 12: 366, 13: 102, 14: 96, 15: 24, 16: 18}
>>>
```
