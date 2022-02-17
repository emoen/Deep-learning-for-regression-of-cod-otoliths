## Exploratory Data Analysis on test-set predictions by the models

### 10-fold training - testset 10% 
| CNN-config | MSE, ACC*  | MSE, ACC** | model error | summary statistics |  pred. error | pred. residuals | residuals misclassificaiton | residual class s.t. rounding error *** |
| - | - | - | - | - | - | - | - | - | 
| EffNetV2 Medium, middle expo.| 0.724<br/> 0.292 | 0.724<br/> 0.295 | <img src="manuscript/eda/EFFNetV2_m_middle_mse/model.png" width="100%" height="100%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/summary.png" width="100%" height="100%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/boxplot_pr_age.png" width="100%" height="100%" >  | <img src="manuscript/eda/EFFNetV2_m_middle_mse/boxplot_residual.png" width="100%" height="100%" > | <img src="manuscript/eda/EFFNetV2_m_middle_mse/misclassification.png" width="100%" height="100%" >| [0.5, 1.5):135 <br/> [1.5, \inf):7 <br> sum:142 |
| EffNet B6, min expo| 0.734<br/> 0.272 | 0.740<br/> 0.268 | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/model.png" width="50%" height="50%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/summary.png" width="50%" height="50%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/boxplot_pr_age.png" width="50%" height="50%" >| <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/boxplot_residual.png" width="50%" height="50%" > | <img src="manuscript/eda/tf_EFFNetB6_groupkfold_stdScalar_10_test_min/misclassification.png" width="50%" height="50%" > | [0.5, 1.5):135 <br/> [1.5, \inf):7 <br> sum:137 |
 
\* Mean MSE, and ACCuracy across 10 models from the 10-fold split <br/>
\** Mean MSE, ACCuracy across 10 models but excluding max/min predictions on each image in the test set
\*** errors larger than 0.5 results in a 1 year misclassification error, 1.5 results in 2 years missclassification error and so on
