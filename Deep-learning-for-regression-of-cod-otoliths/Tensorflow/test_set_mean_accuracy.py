import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

t = [None] * 10
for i in range(0,10):
  t[i] = pd.read_csv("test_set_"+str(i)+".csv")
  
for i in range(0,10):
  print(accuracy_score(t[i].y_pred_test_rounded, t[i].y_true))

for i in range(0,10):
  print(mean_squared_error(t[i].y_pred_test, t[i].y_true))   
  
tmp = t[0].y_pred_test
for i in range(1,10):
  tmp += tmp[i].y_pred_test

tmp = tmp.values/10.0
print(accuracy_score(tmp.rounded().astype("int"), t[0].y_true))
print(mean_squared_error(tmp, t[0].y_true))
