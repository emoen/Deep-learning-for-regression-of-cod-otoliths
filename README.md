# Deep learning for regression of cod otoliths



| Middle Exposure | Min Exposure | Max Exposure |
| - | - | - |
| <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0457_2016_70021.JPG" width="50%" height="50%" > | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0458_2016_70021.JPG" width="50%" height="50%"> | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0459_2016_70021.JPG" width="50%" height="50%"> |
| <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0460_2016_70021.JPG" width="50%" height="50%"> | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0461_2016_70021.JPG" width="50%" height="50%"> | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/IMG_0462_2016_70021.JPG" width="50%" height="50%"> |

The data set consists of 6 images of each otolith. First 3 images with 3 different exposures. Then a rotation of 180 degrees and another 3 images.
The project investigates both which architectures in the EfficientNet family is best to age the otoliths as a regression task using Mean Squared Error (MSE), and which images in the protocol produces the best images. The images to investigate is min, middle and max exposure aswell as a 9 channel image containing all the 3 images. 
The data set consists of 5150 images from age 1 to 13.

|<img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/2013_70174_Nr06_age09_IMG_0031_32_33.png" width="50%" height="50%" >|
|:--:| 
| *Figure 1.:An example of a 9 channel image at the bottom right, where the image is represented as the expectation of the 3 images across channels.:* |



Findings so far: B5, and B6 is better than V2 large, training on more data is better. Testing on same size as training set gives higher accuracy than on test set size, as described in the paper. Training on 3 images (9 channels) with different lighting is better than any one lighting.

For full-precision results: [readme details](https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/README_detailed.md)

Exporatory Data Analysis (EDA) on the models: [readme eda](https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/README_eda.md)

Summary of best results on training on cod otoliths compared to other projects: 
| Species              | Predict    |validLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f |
| ---------------------| -----------|--------|------|-------|-----|-----|---------|---------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124  |0.262|x    |8875     | linear  |
| Salmon               | sea age    | -"-    |0.239 |0.141  |0.822|x    |9073     | linear  |
| Salmon B4            | river age  |0.359   |0.359 |19.58  |0.618|x    |6246     | linear  |
| Cod B5               | age        |0.277   |0.8796| -     |0.744|x    |5150     | linear  |

### 5-fold training - training set: 68%, validation set: 17%, test set 15%
| NN-config              | fold-1 (mse, acc) | fold-2 | fold-3  | fold-4 | fold-5 | mean MSE | mean ACC  | datset size | 
| -----------------------| ------------------|--------|---------|--------|--------|----------|-----------|-------------|
| B4                     | 0.486, 0.649| 0.469, 0.670| 0.482, 0.663 | 0.488, 0.649| 0.473, 0.658 | 0.422 |0.697 |  5150 |
| B4,standardScalar on target, StratifiedKFold| 0.490, 0.644 |0.535, 0.623 | 0.497, 0.661 | 0.457, 0.698 | 0.513, 0.651 |0.426 |0.701 |  5150 |  
| B4,standardScalar on target, StratifiedKFold, pretraining on salmon-scales 20 epochs |0.469, 0.663 | 0.469, 0.685 | 0.513, 0.651 | 0.474, 0.681 | 0.479, 0.650 | 0.433 | 0.689 |  5150|
| B5,standardScalar on target, StratifiedKFold | 0.435, 0.667 | 0.447, 0.683 | 0.451, 0.677 | 0.431, 0.675 | 0.441, 0.692 | 0.401 | 0.707 |  5150|


### 10-fold training, training set: 81% validation set: 9%, test set 10%  - 515 images - no augmentaion on EfficientNetV2
| NN-config (mse, acc)   | 1  | 2 | 3  | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean MSE | mean ACC  | datset size | 
| -----------------------| --------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
| B4 with B5 img size,standardScalar on target, StratifiedKFold       | 0.320,<br/>0.699| 0.318,<br/>0.689| 0.306,<br/>0.687|0.313,<br/>0.683|0.322,<br/>0.689|0.314,<br/>0.701|0.315,<br/>0.697|0.316,<br/>0.668|0.306,<br/>0.689|0.302,<br/>0.724|0.277|0.728|5150|
|B5,standardScalar on target, StratifiedKFold |0.324,<br/>0.718|0.322,<br/>0.691|0.325,<br/>0.693|0.336,<br/>0.668|0.291,<br/>0.736|0.314,<br/>0.707|0.320,<br/>0.662|0.331,<br/>0.683|0.3298,<br/>0.695|0.317,<br/>0.687|0.277|0.744|5150|
|B6,standardScalar on target, StratifiedKFold, min |0.325,<br/>0.683|0.329,<br/>0.685|0.334,<br/>0.664|0.293,<br/>0.724|0.312,<br/> 0.707|0.290,<br/>0.709|0.320,<br/>0.693|0.306,<br/>0.693|0.276,<br/>0.720|0.300,<br/>0.689|0.272|0.734|5150|
|B6,standardScalar on target, StratifiedKFold, middle |0.323,<br/>0.685|0.301,<br/>0.699|0.312,<br/>0.676|0.268,<br/>0.736|0.294,<br/>0.728|0.266,<br/>0.720|0.309,<br/>0.680|0.311,<br/>0.693|0.278,<br/>0.720|0.289,<br/>0.711|0.262|0.744|5150|
| EfficientNetV2-m baseline      | 0.436,<br/> 0.586 | 0.329,<br/> 0.676 | 0.336,<br/> 0.678 |0.374,<br/> 0.637|0.392,<br/>0.625|0.361,<br/>0.654|0.344,<br/> 0.660|0.375,<br/> 0.639|0.322,<br/>0.658|0.328,<br/>0.666|0.331|0.670|5150|
| EfficientNetV2-l  baseline     | 0.363,<br/> 0.65 | 0.360,<br/> 0.652|0.435,<br/>0.641|0.344,<br/>0.670|0.381,<br/>0.631|0.352,<br/>0.664|0.377,<br/>0.648|0.355,<br/>0.658|0.339,<br/>0.656|0.350,<br/>0.658|0.348|0.676|5150|
| EfficientNetV2-m exposure="middle" |0.397, 0.608 |0.374,<br/>0.652|0.356,<br/>0.660|0.384,<br/>0.627|0.350,<br/>0.654|0.337,<br/>0.668|0.326,<br/>0.658|0.365,<br/>0.621|0.353,<br/>0.664|0.335,<br/>0.664|0.336|0.643|5150|
| EfficientNetV2-m exposure="max"| 0.455,<br/>0.588|0.369,<br/>0.652|0.412,<br/>0.610|0.351,<br/>0.645|0.343,<br/>0.680|0.413,<br/>0.604|0.358,<br/>0.658|0.365,<br/>0.649|0.441,<br/>0.581|0.354,<br/>0.654|0.360|0.652|5150|
| EfficientNetV2-m exposure="max" without mixed precision (amp.GradScaler())|0.456,<br/> 0.579|0.396,<br/>0.639|0.387,<br/>0.631|0.372,<br/>0.643|0.395,<br/>0.635|0.381,<br/>0.631|0.369,<br/>0.635|0.447,<br/>0.579|0.433,<br/>0.610|0.3631,<br/>0.633|0.383|0.627|5150|
| EfficientNetV2-l MLP(256,32,1)| 0.363,<br/>0.664|0.378,<br/>0.654|0.405,<br/>0.662|0.342,<br/>0.660|0.393,<br/>0.654|0.370,<br/> 0.668|0.446,<br/>0.639|0.344,<br/>0.668|0.333,<br/> 0.666|0.363,<br/>0.656|0.358|0.662|5150|

### 10-fold training - testset 10% on EffNetV2 with albumenation (-90,90) rotation
| NN-config (val_mse,val_acc),(mse, acc)| 1  | 2 | 3  | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean MSE | mean ACC  | datset size |
| -----------------------| ------------------|--------|---------|--------|--------|--------|--------|--------|--------|-------- |----------|-----------|-------------|
|EfficientNetV2-m exposure="max"|0.371,<br/>0.662|0.456,<br/>0.623|0.355,<br/>0.645|0.405,<br/>0.614|0.886,<br/>0.441|0.481,<br/>0.623|0.370,<br/>0.654|0.459,<br/>0.633|0.803,<br/>0.521|0.595,<br/>0.6|0.381|0.658|5150|
|EfficientNetV2-m exposure="max" MLE savepoints|0.390,<br/>0.635|0.398,<br/>0.619|0.346,<br/>0.650|0.392,<br/>0.647|0.394,<br/>0.619|0.365,<br/>0.662|0.329,<br/>0.672|0.459,<br/>0.581|0.448,<br/>0.614|0.381,<br/>0.645|0.402|0.650|5150|
|EfficientNetV2-m MLP(256,32,1) exposure="middle" MLE savepoints |0.321,<br/> 0.687|0.377,<br/>0.676|0.332,<br/>0.683|0.285,<br/>0.711|0.285,<br/>0.701|0.325,<br/>0.705|0.311,<br/>0.699|0.348,<br/>0.683|0.295,<br/>0.699|0.373,<br/>0.660|0.292|0.724|5150|
|EfficientNetV2-l MLP(256,32,1) MLE savepoints, same test image size, middle |0.301,<br/>0.697|0.281,<br/>0.734|0.299,<br/>0.691|0.318,<br/>0.670|0.282,<br/>0.718|0.305,<br/>0.699|0.280,<br/>0.726|0.334,<br/>0.682|0.300,<br/>0.705|0.310,<br/>0.703|0.280|0.718|5150|
| EfficientNetV2-l MLP(256,32,1) Reload weights, middle| 0.322,<br/>0.666|0.3455,<br/>0.636|0.428,<br/>0.596|||||||||||5150|
| EfficientNetV2-l MLP(256,32,1) Reload weights test_img size 480x480, middle|0.336,<br/>0.656|0.331,<br/>0.645|0.324,<br/>0.648|||||||||||5150|
| EfficientNetV2-l MLP(256,32,1) 9 channels, mse savepoints,test_img=384 |0.292,<br/>0.709|0.289,<br/>0.707|0.289,<br/> 0.705|0.326,<br/>0.707|0.307,<br/>0.715|0.327,<br/>0.693|0.283,<br/>0.707|0.29997,<br/>0.718|0.335,<br/>0.697|0.295,<br/>0.709|0.281|0.717|5150|
|RexNet|0.388,<br/>0.616 |0.446,<br/>0.561|0.379,<br/>0.61||||||||||5150|

### Age distribution of data set of 5150 images

```{1: 382, 2: 522, 3: 509, 4: 624, 5: 805, 6: 540, 7: 544, 8: 477, 9: 327, 10: 217, 11: 122, 12: 55, 13: 26}```
| <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/age_distribution.png" width="50%" height="50%"> |
|:--:| 
| *Figure 2.:Age distribution of cod otoliths:* |

### Test-set age distribution of data set of 515 images

```{1: 41, 2: 59, 3: 52, 4: 60, 5: 90, 6: 52, 7: 55, 8: 47, 9: 23, 10: 19, 11: 13, 12: 2, 13: 2}```
|<img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/imgs/age_distribution_test.png" width="50%" height="50%"> |
|:--:| 
| *Figure 3.:Test set age distribution of cod otoliths:* |
