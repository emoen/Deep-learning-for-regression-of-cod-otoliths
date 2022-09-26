models = c(rep('B4_min',10),rep('B4_mid',10), rep('B4_max',10),
           rep('B5_min',10),rep('B5_mid',10), rep('B5_max',10),
           rep('B6_min',10),rep('B6_mid',10), rep('B6_max',10),
           rep('m_min',10),rep('m_mid',10), rep('m_max',10), rep('m_all',10),
           rep('l_min',10),rep('l_mid',10), rep('l_max',10), rep('l_all', 10))
setwd('C:/prosjekt/cod-otoliths/manuscript/eda/anova' ) 
acc_weights= read.csv('acc_B4_to_large.txt')
x = models
y = acc_weights$Acc
df = data.frame(x, y)
#plot(y ~ x, data = df)
ggplot(df, aes(x=x, y=y)) + geom_boxplot(fill='steelblue')
model.anova = aov(y~x, data=df)
summary(model.anova)