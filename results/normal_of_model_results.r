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

> models = c(0.277,0.277,0.272,0.273,0.280,0.285,0.273,0.262,0.292,0.275,0.291,0.359,0.305,0.290,0.286,0.273,0.271)
> sd(models) 
[1] 0.02174434
> mean(models)
[1] 0.2847647
> min(models)
[1] 0.262
> max(models)
[1] 0.359
> pnorm(max(models), mean(models), sd(models))
[1] 0.9996799
> pnorm(min(models), mean(models), sd(models))
[1] 0.1475669
