# cod-otoliths

For full-precision results: [readme details](https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/README_detailed.md)

Summary of best results on training on cod otoliths compared to other projects: 
| Species              | Predict    |validLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f |
| ---------------------| -----------|--------|------|-------|-----|-----|---------|---------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124  |0.262|x    |8875     | linear  |
| Salmon               | sea age    | -"-    |0.239 |0.141  |0.822|x    |9073     | linear  |
| Salmon B4            | river age  |0.359   |0.359 |19.58  |0.618|x    |6246     | linear  |
| Cod B5               | age        |0.277   |0.8796| -     |0.744|x    |5150     | linear  |

### 5-fold training - testset 15%
| NN-config              | fold-1 (mse, acc) | fold-2 | fold-3  | fold-4 | fold-5 | mean MSE | mean ACC  | datset size | 
| -----------------------| ------------------|--------|---------|--------|--------|----------|-----------|-------------|
| B4                     | 0.486, 0.649| 0.469, 0.670| 0.482, 0.663 | 0.488, 0.649| 0.473, 0.658 | 0.422 |0.697 |  5150 |
| B4,standardScalar on target, StratifiedKFold| 0.490, 0.644 |0.535, 0.623 | 0.497, 0.661 | 0.457, 0.698 | 0.513, 0.651 |0.426 |0.701 |  5150 |  
| B4,standardScalar on target, StratifiedKFold, pretraining on salmon-scales 20 epochs |0.469, 0.663 | 0.469, 0.685 | 0.513, 0.651 | 0.474, 0.681 | 0.479, 0.650 | 0.433 | 0.689 |  5150|
| B5,standardScalar on target, StratifiedKFold | 0.435, 0.667 | 0.447, 0.683 | 0.451, 0.677 | 0.431, 0.675 | 0.441, 0.692 | 0.401 | 0.707 |  5150|


### 10-fold training - testset 10%
| NN-config (mse, acc)   | fold-1  | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size | 
| -----------------------| --------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
| B4 with B5 img size,standardScalar on target, StratifiedKFold       | 0.320,0.699| 0.318,0.689| 0.306,0.687|0.313,0.683|0.322,0.689|0.314,0.701|0.315,0.697|0.316,0.668|0.306,0.689|0.302,0.724|0.277|0.728|5150|
|B5,standardScalar on target, StratifiedKFold |0.324,0.718|0.322,0.691|0.325,0.693|0.336,0.668|0.291,0.736|0.314,0.707|0.320,0.662|0.331,0.683|0.3298,0.695|0.317,0.687|0.277|0.744|5150|
|B6,standardScalar on target, StratifiedKFold |0.325,0.683|0.329,0.685|0.334,0.664|0.293,0.724|0.312, 0.707|0.290,0.709|0.320,0.693|0.306,0.693|0.276,0.720|0.300,0.689|0.272|0.734|5150|
| EfficientNetV2-m baseline      | 0.436, 0.586 | 0.329, 0.676 | 0.336, 0.678 |0.374, 0.637|0.392,0.625|0.361,0.654|0.344, 0.660|0.375, 0.639|0.322,0.658|0.328,0.666|0.331|0.670|5150|
| EfficientNetV2-l  baseline     | 0.363, 0.65 | 0.360, 0.652|0.435,0.641|0.344,0.670|0.381,0.631|0.352,0.664|0.377,0.648|0.355,0.658|0.339,0.656|0.350,0.658|0.348|0.676|5150|
| EfficientNetV2-m exposure="middle" |0.397, 0.608 |0.374,0.652|0.356,0.660|0.384,0.627|0.350,0.654|0.337,0.668|0.326,0.658|0.365,0.621|0.353,0.664|0.335,0.664|0.336|0.643|5150|
| EfficientNetV2-m exposure="max"| 0.455,0.588|0.369,0.652|0.412,0.610|0.351,0.645|0.343,0.680|0.413,0.604|0.358,0.658|0.365,0.649|0.441,0.581|0.354,0.654|0.360|0.652|5150|
| EfficientNetV2-m exposure="max" without mixed precision (amp.GradScaler())|0.456, 0.579|0.396,0.639|0.387,0.631|0.372,0.643|0.395,0.635|0.381,0.631|0.369,0.635|0.447,0.579|0.433,0.610|0.3631,0.633|0.383|0.627|5150|
| EfficientNetV2-l MLP(256,32,1)| 0.363,0.664|0.378,0.654|0.405,0.662|0.342,0.660|0.393,0.654|0.370, 0.668|0.446,0.639|0.344,0.668|0.333,0.666|0.363,0.656|0.358|0.662|5150|

### 10-fold training - testset 10% on EffNetV2 with albumenation (-90,90) rotation
| NN-config (val_mse,val_acc),(mse, acc)| fold-1  | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size |
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
|EfficientNetV2-m exposure="max"|0.371,0.662|0.456,0.623|0.355,0.645|0.405,0.614|0.886,0.441|0.481,0.623|0.370,0.654|0.459,0.633|0.803,0.521|0.595,0.6]|0.381|0.658|5150|
|EfficientNetV2-m exposure="max" MLE savepoints|0.390,0.635|0.398,0.619|0.346,0.650|0.392,0.647|0.394,0.619|0.365,0.662|0.329,0.672|0.459,0.581|0.448,0.614|0.381,0.645|0.402|0.650|5150|
|EfficientNetV2-l MLP(256,32,1) MLE savepoints, same test image size |0.301,0.697|0.281,0.734|0.299,0.691|0.318,0.670|0.282,0.718|0.305,0.699|0.280, 0.726|0.334,0.682|0.300,0.705|0.310,0.703|0.280|0.718|5150|
| EfficientNetV2-l MLP(256,32,1) Reload weights| 0.322, 0.666|0.3455, 0.636|0.428,0.596|||||||||||5150|
| EfficientNetV2-l MLP(256,32,1) Reload weights test_img size 480x480|0.336,0.656|0.331,0.645|0.324,0.648|||||||||||5150|
|RexNet| ||||||||||||5150|

### Test-set age distribution

```{1: 41, 2: 59, 3: 52, 4: 60, 5: 90, 6: 52, 7: 55, 8: 47, 9: 23, 10: 19, 11: 13, 12: 2, 13: 2}```

### Age distribution - of dataset
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

