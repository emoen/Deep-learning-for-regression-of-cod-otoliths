models = c(rep('B4_min',10),rep('B4_mid',10), rep('B4_max',10),
           rep('B5_min',10),rep('B5_mid',10), rep('B5_max',10),
           rep('B6_min',10),rep('B6_mid',10), rep('B6_max',10),
           rep('m_min',10),rep('m_mid',10), rep('m_max',10), 
           rep('l_min',10),rep('l_mid',10), rep('l_max',10))
exposure = c("min", "middle", "max")

acc =     c(72.8, 71.5, 70.9, 74.4, 73.4, 73.2, 73.4, 74.4, 71.5, 74.0, 72.4, 71.3, 72.0, 72.8, 72.4)
network = c(1,1,1,2,2,2,3,3,3,4,4,4,5,5,5)     #b4,b5,b6,m,l
exposure =c(1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3) #min,middle, max


dataf = data.frame(acc)
#dataf$network =network
#dataf$exposure=exposure
dataf$network = as.factor(network) #, levels = seq(1:5), labels = c("B4", "B5", "B6", "m", "l"))
dataf$exposure = as.factor(exposure) #, levels = seq(1:3), labels=c("min","mid", "max"))


res.aov2 <- aov(dataf$acc ~ dataf$network + dataf$exposure, data = dataf)
summary(res.aov2)
#Df Sum Sq Mean Sq F value Pr(>F)  
#dataf$network   4  6.409  1.6023   2.369 0.1391  
#dataf$exposure  2  5.649  2.8247   4.176 0.0573 .
#Residuals       8  5.411  0.6763                 
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1                    

# Box plot with two factor variables
boxplot(dataf$acc ~ dataf$network + dataf$exposure, data = twoway, frame = FALSE, 
        col = c("#00AFBB", "#E7B800"), ylab="Tooth Length")
# Two-way interaction plot
interaction.plot(x.factor = dataf$network , trace.factor = dataf$exposure, 
                 response = dataf$acc, fun = mean, 
                 type = "b", legend = TRUE, 
                 xlab = "network", ylab="Acc",
                 pch=c(1,19), col = c("#00AFBB", "#E7B800"))


# With exposure on x-axis
data2 = dataf
library(car)
data2$exposure = recode(data2$exposure, '1 = "Min"; 2 = "Middle"; 3 = "Max"')
data2$network = recode(data2$network, '1 = "B4"; 2 = "B5"; 3 = "B6"; 4 = "Medium"; 5="Large"')
x1 = factor(data2$network, levels=c("B4","B5","B6","Medium", "Large"))

interaction.plot(x.factor = data2$exposure, 
                 trace.factor = x1, 
                 response = data2$acc, 
                 fun = mean, 
                 type = "b", 
                 lty=1:2,lwd=1:4,
                 fixed = TRUE,
                 legend = TRUE, 
                 xlab = "Exposure",
                 ylab = "Accuracy",
                 pch = c(1,19), 
                 col = c("red", "blue", "gray", "pink", "green", "orange"),
                 trace.label = "Network")

