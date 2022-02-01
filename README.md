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
| NN-config              | fold-1 (mse, acc)(max, mean, min) | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size | 
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
| B4 with B5 img size,standardScalar on target, StratifiedKFold       | 0.320,0.699| 0.318,0.689| 0.306,0.687|0.313,0.683|0.322,0.689|0.314,0.701|0.315,0.697|0.316,0.668|0.306,0.689|0.302,0.724|0.277|0.728|5150|
|B5,standardScalar on target, StratifiedKFold |0.324,0.718|0.322,0.691|0.325,0.693|0.336,0.668|0.291,0.736|0.314,0.707|0.320,0.662|0.331,0.683|0.3298,0.695|0.317,0.687|0.277|0.744|5150|
|B6,standardScalar on target, StratifiedKFold |0.325,0.683|0.329,0.685|0.334,0.664|0.293,0.724|0.312, 0.707|0.290,0.709|0.320,0.693|0.306,0.693|0.276,0.720|0.300,0.689|0.272|0.734|5150|
| EfficientNetV2-m baseline      | 0.436, 0.586 | 0.329, 0.676 | 0.336, 0.678 |0.374, 0.637|0.392,0.625|0.361,0.654|0.344, 0.660|0.375, 0.639|0.322,0.658|0.328,0.666|0.331|0.670|5150|
| EfficientNetV2-l  baseline     | 0.363, 0.65 | 0.360, 0.652|0.435,0.641|0.344,0.670|0.381,0.631|0.352,0.664|0.377,0.648|0.355,0.658|0.339,0.656|0.350,0.658|0.348|0.676|5150|
| EfficientNetV2-m exposure="middle" |0.397, 0.608 |0.374,0.652|0.356,0.660|0.384,0.627|0.350,0.654|0.337,0.668|0.326,0.658|0.365,0.621|0.353,0.664|0.335,0.664|0.336|0.643|5150|
| EfficientNetV2-m exposure="max"| 0.455,0.588|0.369,0.652|0.412,0.610|0.351,0.645|0.343,0.680|0.413,0.604|0.358,0.658|0.365,0.649|0.441,0.581|0.354,0.654|0.360|0.652|5150|
| EfficientNetV2-m exposure="max" without mixed precision (amp.GradScaler())|[0.455899366829671, 0.5786407766990291][11.624638557434082,5.510402609075157,1.029883861541748]|[0.3955207692059546,0.6388349514563106]|[0.3870887313689006,0.6310679611650486]|[0.37190759645808213,0.6427184466019418]|[0.3954467958043927,0.6349514563106796]|[0.3806486774090985,0.6310679611650486]|[0.3694079017430422,0.6349514563106796]|[0.4474771832160993,0.5786407766990291]|[0.4329707149162454,0.6097087378640776]|[0.3630835861605383,0.6330097087378641]|0.38274013996009026|0.6271844660194175|5150|
| EfficientNetV2-l MLP(256,32,1)| [0.3629718165260899,0.6640776699029126],[11.326478958129883,5.176436455967357,0.7270090579986572]|[0.3778153902815894,0.654368932038835],[11.756083488464355,5.078827020496998,0.7515597939491272]|[0.40490400097933393,0.6621359223300971],[11.678875923156738,5.0282682075083835,0.6951654553413391]|[0.3422837114291397,0.6601941747572816],[11.539137840270996,5.027335057906734,0.7135916948318481]|[0.3930839488501394,0.654368932038835],[11.62857723236084,5.065871095078663,0.8129523992538452]|[0.36973989926423256, 0.6679611650485436],[11.52272891998291, 5.0117800284357905,0.7823890447616577]|[0.44649657125737163,0.6388349514563106],[11.628984451293945,5.012184920820218,0.45771318674087524]|[0.3442822287250434,0.6679611650485436],[11.645936965942383,5.125562575951363,0.709739625453949]|[0.3326264110015347,0.6660194174757281],[11.753767967224121,5.1002887210799654,0.7644101977348328]|[0.3626785866047162,0.6563106796116505],[11.611091613769531,5.029804865249153,0.7668406367301941]|0.3578598940821187|0.6621359223300971|5150|

### 10-fold training - testset 10% on EffNetV2 with albumenation (-90,90) rotation
| NN-config              | fold-1 (val_mse,val_acc),(mse, acc) | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size |
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
|EfficientNetV2-m exposure="max"|[0.37055935391494754,0.6621359223300971]|[0.45588673297899246,0.6233009708737864]|[0.3547869452993763,0.6446601941747573]|[0.4045900893110558,0.6135922330097088]|[0.88552916,0.4406047516198704]|[0.48096464346891804,0.6233009708737864]|[0.37039235639865814,0.654368932038835]|[0.45940533538042655,0.6330097087378641]|[0.8034557,0.5205183585313174]|[0.5949553335231461,0.6]|0.3811391933261682|0.658252427184466|5150|
|EfficientNetV2-m exposure="max" MLE savepoints|0.39006542761913027,0.6349514563106796|0.3975690467732811,0.6194174757281553|0.3456209812726769,0.6504854368932039|0.392271179482109,0.6466019417475728|0.3944861143057912,0.6194174757281553|0.36419188526757085,0.6621359223300971|0.32908238390696287,0.6718446601941748|0.4586693277976992,0.5805825242718446|0.4482269191065332,0.6135922330097088|0.38145080447413454,0.6446601941747573|0.40194174757281553|0.6504854368932039|5150|
|EfficientNetV2-l MLP(256,32,1) MLE savepoints, same test image size |[0.31076611375098306,0.6893203883495146]|[0.4524055561189933,0.5941747572815534]||||||||||||
|EfficientNetV2-l MLP(256,32,1) MLE savepoints, same test image size |[0.30097150575771525,0.6970873786407767]|[0.28132935946744175,0.7339805825242719]|[0.29904481883102846,0.6912621359223301|[0.3176863822144107,0.6699029126213593]|[0.28227414303887455,0.7184466019417476]|[0.3049059022718125,0.6990291262135923]|[0.2796412552907852, 0.7262135922330097]|0.33390750726134205,0.6815533980582524]|[0.2998588231157967,0.7048543689320388]|[0.31008770317437595,0.7029126213592233]|0.27972858674427004|0.7184466019417476|5150|

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

