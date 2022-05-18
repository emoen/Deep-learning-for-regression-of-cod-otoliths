> models = c(72.8,74.4,73.4,74.0,72.0,71.5,73.4,74.4,72.4,72.8,70.9,73.2,71.5,71.1,72.4,74.0,72.2)
> sd(models) 
[1] 1.117902
> mean(models)
[1] 72.72941
> min(models)
[1] 70.9
> max(models)
[1] 74.4
> pnorm(max(models), mean(models), sd(models))
[1] 0.9324638
> pnorm(min(models), mean(models), sd(models))
[1] 0.05087082
