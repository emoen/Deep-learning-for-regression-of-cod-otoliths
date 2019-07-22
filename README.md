# cod-otoliths

Predicting age of cod otoliths

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species              | Predict    |testLOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f | classWeights |
| ---------------------| -----------|--------|------|------|-----|-----|---------|---------|--------------|
| Greenland Halibut(1) | age        | x      |2.65  |0.124 |0.262|x    |8875     | linear  | x | 
| Salmon               | sea age    | -"-    |0.239 |0.141 |0.822|x    |9073     | linear  | x |
| Salmon B4            | river age  |0.359   |0.359 |19.58 |0.618|x    |6246     | linear  | x |
| Cod B4               | age        |x   |x |x |x|x    |6330     | linear | x |
