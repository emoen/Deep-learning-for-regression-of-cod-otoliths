## Exploratory Data Analysis on test-set predictions by model

Box-plot of models on the 10-fold predictions on the test-set. The red line is the ensembled accuracy/MSE.

<img src="manuscript/eda/box_plot_models_acc.png" width="50%" height="50%" > <br/>
<img src="manuscript/eda/box_plot_models_mse.png" width="50%" height="50%" >

### 10-fold training - testset 10% 
| CNN-config | MSE, ACC*  |  Box plot model error summary | summary statistics |  prediction error | prediction residual | residuals misclassificaiton** |  
| -  | - | - | - | - | - | - | 
| EffNetV2 Medium, middle expo.| (0.724 0.292) <br/> (0.724, 0.295) | <img src="manuscript/eda/EFFNetV2_m_middle_mse/model.png" width="200%" height="200%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/summary.png" width="200%" height="200%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/boxplot_pr_age.png" width="200%" height="200%" >  | <img src="manuscript/eda/EFFNetV2_m_middle_mse/boxplot_residual.png" width="200%" height="200%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/misclassification.png" width="200%" height="200%" > <br/> [0.5, 1.5):135, [1.5, \inf):7, sum:142 |
| EffNet B6, min expo| (0.734, 0.272) <br/> (0.740, 0.268) | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/model.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/summary.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/boxplot_pr_age.png" width="200%" height="200%" >| <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/boxplot_residual.png" width="200%" height="200%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/misclassification.png" width="150%" height="150%" > <br/> [0.5, 1.5):128, [1.5, \inf):9, sum:137 |
| EffNet B5, min expo| (0.744, 0.277) <br/> (0.736, 0.272) | <img src="manuscript/eda/tf_EFFNetB5_2_groupkfold_stdScalar_10_test_min/model.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB5_2_groupkfold_stdScalar_10_test_min/summary.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB5_2_groupkfold_stdScalar_10_test_min/boxplot_pr_age.png" width="200%" height="200%" >| <img src="manuscript/eda/tf_EFFNetB5_2_groupkfold_stdScalar_10_test_min/boxplot_residual.png" width="200%" height="200%" > | <img src="manuscript/eda/tf_EFFNetB5_2_groupkfold_stdScalar_10_test_min/misclassification.png" width="150%" height="150%" > <br/> [0.5, 1.5):125, [1.5, \inf):7, sum:132 |
| EffNet B4, min expo| (0.728, 0.277) <br/> (0.732, 0.273) | <img src="manuscript/eda/tf_EFFNetB4_groupkfold_stdScalar_10_test2/model.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB4_groupkfold_stdScalar_10_test2/summary.png" width="250%" height="250%" > | <img src="manuscript/eda/tf_EFFNetB4_groupkfold_stdScalar_10_test2/boxplot_pr_age.png" width="200%" height="200%" >| <img src="manuscript/eda/tf_EFFNetB4_groupkfold_stdScalar_10_test2/boxplot_residual.png" width="200%" height="200%" > | <img src="manuscript/eda/tf_EFFNetB4_groupkfold_stdScalar_10_test2/misclassification.png" width="150%" height="150%" > <br/> [0.5, 1.5):132, [1.5, \inf):8, sum:140 |
 
\* First 2 numbers is mean MSE, and ACCuracy across 10 models from the 10-fold split, then the second 2 numbers are mean MSE, ACCuracy across 10 models but excluding max/min predictions on each image in the test set <br/>
\** Numbers at the bottom are errors larger than 0.5 results in a 1 year misclassification error, 1.5 results in 2 years missclassification error and so on

## Outliers

Outliers with an error of more than 1.5 years:
| V2-m, middle | V2-l, all | V2-l, middle | B4, min | B5, min |  B6, min | B6, middle |  | y_true |
|--------------|-----------|--------------|---------|---------|----------|------------|--|--------|
|              |           |              | 13      | 13      | 13       | 13         |  | 8      |
|              |           |              |         |         | 48       |            |  | 6      |
| 71           | 71        | 71           | 71      | 71      | 71       | 71         |  | 7      |
| 92           |           |              |         |         |          |            |  | 13     |
|              |           |              | 270     | 270     |          | 270        |  | 10     |
| 279          | 279       | 279          | 279     | 279     | 279      | 279        |  | 8      |
|              | 312       | 312          |         |         |          |            |  |        |
|              |           | 320          | 320     |         |          |            |  | 7      |
| 362          | 362       | 362          | 362     | 362     | 362      | 362        |  | 7      |
| 342          | 342       | 342          | 342     | 342     | 342      | 342        |  | 13     |
| 369          | 369       | 369          | 369     |         | 369      | 369        |  | 10     |
|              |           | 393          |         |         | 393      | 393        |  |        |
| 423          | 423       | 423          |         |         |          |            |  | 8      |
|              |           |              |         | 444     |          |            |  | 9      |
|              |           |              |         |         | 502      | 502        |  | 11     |
|              |           |              |         |         |          |            |  |        |
| 7            | 7         |              | 8       | 7       | 9        | 9          |  |        |

## Largest outliers with predictions and ground-truth
|  Idx| V2-m, middle | V2-l, all | V2-l, middle | B4, min | B5, min |  B6, min | B6, middle |  | y_true |
|-----|--------------|-----------|--------------|---------|---------|----------|------------|--|--------|
| 13  |              |           |              | 9.79    | 9.64    | 9.74     | 9.58       |  | 8      |
| 48  |              |           |              |         |         | 7.6      |            |  | 6      |
| 71  | 4.96         | 4.98      | 4.94         | 5.14    | 4.79    | 5.06     | 5.12       |  | 7      |
| 92  | 10.95        |           |              |         |         |          |            |  | 13     |
| 270 |              |           |              | 11.66   | 11.71   |          | 11.53      |  | 10     |
| 279 | 9.93         | 9.79      | 9.75         | 9.89    | 9.69    | 9.67     | 9.7        |  | 8      |
| 312 |              | 9.42      | 9.38         |         |         |          |            |  |        |
| 320 |              |           | 5.44         | 5.47    |         |          |            |  | 7      |
| 362 | 5.11         | 5.14      | 5.23         | 5.11    | 5.29    | 5.24     | 5.15       |  | 7      |
| 342 | 10.35        | 10.6      | 10.61        | 11.05   | 10.75   | 10.69    | 10.84      |  | 13     |
| 369 | 8.17         | 8.13      | 8.23         | 8.24    |         | 7.85     | 8.29       |  | 10     |
| 393 |              |           | 10.53        |         |         | 10.75    | 10.83      |  |        |
| 423 | 5.39         | 5.69      | 5.43         |         |         |          |            |  | 8      |
| 444 |              |           |              |         | 10.95   |          |            |  | 9      |
| 502 |              |           |              |         |         | 9.4      | 9.43       |  | 11     |

## Images of most common outliers: {13, 71, 270, 342, 360, 369}

| Index | 13 | 71 | 270 | 342 | 360 | 269 |
| - | - | - | - | - | - | - |
| Image | <img src="manuscript/eda/outliers/IMG_0284_13.JPG" width="100%" height="100%" > | <img src="manuscript/eda/outliers/IMG_0230_71.JPG" width="100%" height="100%" > | <img src="manuscript/eda/outliers/IMG_0104_270.JPG" width="100%" height="100%" > | <img src="manuscript/eda/outliers/IMG_0044_342.JPG" width="100%" height="100%" > | <img src="manuscript/eda/outliers/IMG_0086_360.JPG" width="100%" height="100%" > | <img src="manuscript/eda/outliers/IMG_0122_369.JPG" width="100%" height="100%" > |


