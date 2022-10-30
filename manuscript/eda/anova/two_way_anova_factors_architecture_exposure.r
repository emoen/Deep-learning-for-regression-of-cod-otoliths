models = c(rep('B4_min',10),rep('B4_mid',10), rep('B4_max',10),
           rep('B5_min',10),rep('B5_mid',10), rep('B5_max',10),
           rep('B6_min',10),rep('B6_mid',10), rep('B6_max',10),
           rep('m_min',10),rep('m_mid',10), rep('m_max',10), rep('m_all',10),
           rep('l_min',10),rep('l_mid',10), rep('l_max',10), rep('l_all', 10))
exposure = c("min", "middle", "max")
setwd('C:/prosjekt/cod-otoliths/manuscript/eda/anliova' ) 
acc_weights= read.csv('acc_B4_to_large.txt')

dataf = data.frame(models)
dataf$models <- factor(models, levels = seq(1:5), labels = c("B4", "B5", "B6", "m", "l"))
dataf$exposure = factor(exposure, levels = seq(1:3), labels=c("min","mid", "max"))
dataf$acc = 