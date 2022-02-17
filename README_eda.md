## Exploratory Data Analysis on test-set predictions by the models

### 10-fold training - testset 10% 
| CNN-config | MSE, ACC  | MSE, ACC exl min/max | model error | summary statistics | model error exl.min-max | pred. error | pred. residuals | residuals misclassificaiton |
| - | - | - | - | - | - | - | - | - |
| EffNetV2 Medium, middle expo.| 0.724<br/> 0.292 | 0.724<br/> 0.295 | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/eda/EFFNetV2_m_middle_mse/model.png" width="50%" height="50%" > | count	515.000000
mean	0.011365 <br/>
std	0.540606 <br/>
min	-1.939247 <br/>
25%	-0.284032 <br/>
50%	-0.026677 <br/>
75%	0.259814 <br/>
max	2.646767 <br/> | | | |
| EffNet B6, min expo| 0.734<br/> 0.272 | 0.740<br/> 0.268 | <img src="https://github.com/emoen/Deep-learning-for-regression-of-cod-otoliths/blob/master/manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/model.png" width="50%" height="50%" > | | | | | |
 

