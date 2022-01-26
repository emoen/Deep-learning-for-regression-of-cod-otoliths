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
| NN-config              | fold-1 (mse, acc) | fold-2 | fold-3  | fold-4 | fold-5 | mean MSE | mean ACC  | datset size |  |
| -----------------------| ------------------|--------|---------|--------|--------|----------|-----------|-------------|--|
| B4                     | [0.48629727959632874, 0.6488250494003296]|[0.4687076508998871, 0.6697127819061279]|[0.4820464551448822, 0.6631853580474854] |[0.4878818988800049, 0.6488250494003296]|[0.47346818447113037, 0.6579634547233582] |0.4216341924334377 |0.6971279373368147 |  5150       |  | 
| B4,standardScalar on target, StratifiedKFold| [0.4904178682121655, 0.643603133159269] |[0.5347786293189214, 0.6227154046997389] | [0.4968175576778982, 0.660574412532637] | [0.4573790357033823, 0.6984334203655352] | [0.5134489160772718, 0.6514360313315927] |0.4257378207634962 |0.7010443864229765  |  5150       |  | 
| B4,standardScalar on target, StratifiedKFold, pretraining on salmon-scales 20 epochs |[0.46910862052397906, 0.6631853785900783] | [0.4689538583672292, 0.685378590078329] | [0.5130625894583206, 0.6514360313315927] | [0.4741178483435903, 0.6814621409921671] |[0.47853572666031796, 0.6501305483028721] | 0.43275226756928326 | 0.6892950391644909 |  5150       |  | 
| B5,standardScalar on target, StratifiedKFold | [0.4353308031328409, 0.667098445595855 ] | [0.44663970124877767, 0.6826424870466321] | [0.45198405952545945, 0.677461139896373] | [0.430863676385045, 0.6748704663212435] | [0.44126310267337826, 0.6917098445595855] | 0.40109202928591997 | 0.7072538860103627 |  5150       |  |



### 10-fold training - testset 10%
| NN-config              | fold-1 (mse, acc)(max, mean, min) | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size | 
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
| B4 with B5 img size,standardScalar on target, StratifiedKFold       | [0.3197536160088654,0.6990291262135923]| [0.31845087610760114,0.6893203883495146]| [0.30601433207216255,0.6873786407766991]|[0.31335182256242056,0.683495145631068]|[0.32232335618776714,0.6893203883495146]|[0.31398366708967657,0.7009708737864078]|[0.3147587854708994,0.6970873786407767]|[0.3164655743413814,0.6679611650485436]|[0.30647813064746254,0.6893203883495146]|[0.30178253861722815,0.7242718446601941]|0.27677442836796534|0.7281553398058253|5150|
|B5,standardScalar on target, StratifiedKFold |[0.32392935016259955,0.7184466019417476]|[0.3216571342235008,0.6912621359223301]|[0.32478147278277225,0.6932038834951456]|[0.33645739740161507,0.6679611650485436]|[0.29122819849402387,0.7359223300970874]|[0.3136678521029937,0.7067961165048544]|[0.32013204172139337,0.6621359223300971]|[0.3309105222508796,0.683495145631068]|[0.32977773700250607,0.6951456310679611]|[0.31682926098100384,0.6873786407766991]|0.2770159431240281|0.7436893203883496|5150|
|B6,standardScalar on target, StratifiedKFold |[0.32510835507442315,0.683495145631068]|[0.3290709168395908,0.6854368932038835]|[0.33377248527623243,0.6640776699029126]|[0.29291382688065226,0.7242718446601941]|[0.31181813555346594, 0.7067961165048544]|[0.2902428397533708,0.7087378640776699]|[0.3196961505168636,0.6932038834951456]|[0.3060735895344372,0.6932038834951456]|[0.27629437608294805,0.7203883495145631]|[0.29986697830481673,0.6893203883495146]|0.272170267061415|0.7339805825242719|5150|
| EfficientNetV2-m baseline      | [0.43593598646472737, 0.5864077669902913], [11.656976699829102, 5.509386144332515, 1.0240179300308228] | [0.329003091574722, 0.6757281553398058],[11.689010620117188, 5.1254937412669355, 0.9654581546783447] | [0.33623309593114964, 0.6776699029126214], [11.889851570129395, 5.285319744498985, 0.9146838188171387] |[0.3737735574605572, 0.6368932038834951], [12.078588485717773, 5.409902389535626, 0.9477762579917908]|[0.39238114248926304,0.625242718446602],[11.572626113891602,5.190874819500932,0.9491563439369202]|[0.36080863358942594,0.654368932038835],[11.696592330932617,5.305954855854072,0.8887564539909363]|[0.34378273482522753, 0.6601941747572816],[11.557056427001953, 5.2136024237836445, 0.920559823513031]|[0.37456772209432077, 0.6388349514563106],[11.856122016906738,5.3887680463420535,0.9428082704544067]|[0.32232482847267624,0.658252427184466],[11.902791023254395,5.312223237000623,0.913942813873291]|[0.32846365935074273,0.6660194174757281],[12.20690631866455, 5.246973541292172, 0.9430772662162781]|0.33051156280698374|0.6699029126213593|5150|
| EfficientNetV2-l  baseline     | [0.3631332187061773, 0.654368932038835], [12.01773738861084, 5.310474924439365,0.8895091414451599] | [0.35952411192975386, 0.6524271844660194], [11.634198188781738, 5.277551628663702, 0.8856195211410522]|[0.434872708049315,0.6407766990291263],[11.776787757873535,5.303626792176256,0.9191138744354248]|[0.343894515699138,0.6699029126213593],[11.503342628479004,5.10492112439813,0.8544027209281921]|[0.38062042388052225,0.6310679611650486],[11.66337776184082, 5.360615915928072, 0.8762474060058594]|[0.35183026052933125,0.6640776699029126],[11.736310958862305,5.2492036546318275,0.8503969311714172]|[0.3770195560160116,0.6485436893203883],[11.890049934387207,5.2943437687401635,0.9406237006187439]|[0.3549831283558617,0.658252427184466],[11.888720512390137,5.279939203586393,0.9145044088363647]|[0.33879768520217496,0.6563106796116505],[11.994569778442383,5.31596553163621,0.9251725673675537]|[0.3497640113286335,0.658252427184466],[11.943987846374512,5.207454828845645,0.8693790435791016]|0.3482863627148823|0.6757281553398058|5150|
| EfficientNetV2-m exposure="middle" |[0.39718897058209524, 0.6077669902912621],[11.72133731842041,5.447287622701775,0.9450588226318359] |[0.37383119606970494,0.6524271844660194],[11.627776145935059,5.1725509268566245,1.0156219005584717]|[0.35610648247128274,0.6601941747572816],[11.998383522033691,5.361631473985691,0.8636019229888916]|[0.3840084548166239,0.6271844660194175],[11.898012161254883, 5.398851113064775, 0.9133579134941101]|[0.349523375805845,0.654368932038835],[11.257091522216797,5.246484787255815,0.8743095993995667]|[0.33748664972898557,0.6679611650485436],[11.714927673339844,5.247008727707909,0.9867938160896301]|[0.3256831318805641,0.658252427184466],[11.796988487243652,5.233834440384096,0.9097902178764343]|[0.36527177271722916,0.6213592233009708],[11.842670440673828,5.338287988796975,0.9638106226921082]|[0.35276076129821055,0.6640776699029126],[11.92320728302002,5.212955970555833,0.9720532298088074]|[0.3345734053043579,0.6640776699029126],[12.203539848327637,5.287658938620854,0.8788002729415894]|0.3357187723002123|0.6427184466019418|5150|
| EfficientNetV2-m exposure="max"| [0.4552132929754933,0.5883495145631068],[11.542473793029785,5.508298445905297,1.0317736864089966]|[0.36893291988509574,0.6524271844660194],[11.887615203857422,5.27592860066775,0.8879614472389221]|[0.41200427720744315,0.6097087378640776],[12.223085403442383,5.459770287124856,0.8846766352653503]|[0.3505811724148065,0.6446601941747573],[11.851435661315918,5.365713037101968,0.9331003427505493]|[0.34315351951040934,0.6796116504854369],[11.596426963806152,5.156253770369928,0.919688880443573]|[0.4127972184036007,0.6038834951456311],[11.885137557983398,5.45255562101753,0.9943726062774658]|[0.3575387501974067,0.658252427184466],[11.667533874511719,5.304152626319996,0.9186959862709045]|[0.36465211985087903,0.6485436893203883],[11.978194236755371,5.313120795337899,0.9499825239181519]|[0.44050087972013446,0.5805825242718446],[11.716320037841797,5.491412600151543,0.9512141346931458]|[0.3538976686818368,0.654368932038835],[11.76646900177002,5.304721099427603,0.9214164614677429]|0.3601233019856572|0.6524271844660194|5150|
| EfficientNetV2-m exposure="max" without mixed precision (amp.GradScaler())|[0.455899366829671, 0.5786407766990291][11.624638557434082,5.510402609075157,1.029883861541748]|[0.3955207692059546,0.6388349514563106]|[0.3870887313689006,0.6310679611650486]|[0.37190759645808213,0.6427184466019418]|[0.3954467958043927,0.6349514563106796]|[0.3806486774090985,0.6310679611650486]|[0.3694079017430422,0.6349514563106796]|[0.4474771832160993,0.5786407766990291]|[0.4329707149162454,0.6097087378640776]|[0.3630835861605383,0.6330097087378641]|0.38274013996009026|0.6271844660194175|5150|
| EfficientNetV2-l MLP(256,32,1)| [0.3629718165260899,0.6640776699029126],[11.326478958129883,5.176436455967357,0.7270090579986572]|[0.3778153902815894,0.654368932038835],[11.756083488464355,5.078827020496998,0.7515597939491272]|[0.40490400097933393,0.6621359223300971],[11.678875923156738,5.0282682075083835,0.6951654553413391]|[0.3422837114291397,0.6601941747572816],[11.539137840270996,5.027335057906734,0.7135916948318481]|[0.3930839488501394,0.654368932038835],[11.62857723236084,5.065871095078663,0.8129523992538452]|[0.36973989926423256, 0.6679611650485436],[11.52272891998291, 5.0117800284357905,0.7823890447616577]|[0.44649657125737163,0.6388349514563106],[11.628984451293945,5.012184920820218,0.45771318674087524]|[0.3442822287250434,0.6679611650485436],[11.645936965942383,5.125562575951363,0.709739625453949]|[0.3326264110015347,0.6660194174757281],[11.753767967224121,5.1002887210799654,0.7644101977348328]|[0.3626785866047162,0.6563106796116505],[11.611091613769531,5.029804865249153,0.7668406367301941]|0.3578598940821187|0.6621359223300971|5150|

### 10-fold training - testset 10% on EffNetV2 with albumenation (-90,90) rotation
| NN-config              | fold-1 (val_mse,val_acc),(mse, acc) | fold-2 | fold-3  | fold-4 | fold-5 | fold-6 | fold-7 | fold-8 | fold-9 | fold-10 | mean MSE | mean ACC  | datset size |
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
|EfficientNetV2-m exposure="max"|[0.37055935391494754,0.6621359223300971]|[0.45588673297899246,0.6233009708737864]|[0.3547869452993763,0.6446601941747573]|[0.4045900893110558,0.6135922330097088]|[0.88552916,0.4406047516198704]|[0.48096464346891804,0.6233009708737864]|[0.37039235639865814,0.654368932038835]|[0.45940533538042655,0.6330097087378641]|[0.8034557,0.5205183585313174]|[0.5949553335231461,0.6]|0.3811391933261682|0.658252427184466|5150|
|EfficientNetV2-m exposure="max" MLE savepoints|0.39006542761913027,0.6349514563106796|0.3975690467732811,0.6194174757281553|0.3456209812726769,0.6504854368932039|0.392271179482109,0.6466019417475728|0.3944861143057912,0.6194174757281553|0.36419188526757085,0.6621359223300971|0.32908238390696287,0.6718446601941748|0.4586693277976992,0.5805825242718446|0.4482269191065332,0.6135922330097088|0.38145080447413454,0.6446601941747573|0.40194174757281553|0.6504854368932039|5150|
|EfficientNetV2-m exposure="max" MLE savepoints|[0.4397,0.7349137931034483][0.39006542761913027,0.6349514563106796]|[0.4655,0.7198275862068966],[0.3975690467732811,0.6194174757281553]|[0.38362068,0.7262931034482759],[0.3456209812726769,0.6504854368932039]|||||||||||
|EfficientNetV2-l MLP(256,32,1) MLE savepoints, same test image size |[0.31076611375098306,0.6893203883495146]|[0.4524055561189933,0.5941747572815534]||||||||||||

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

