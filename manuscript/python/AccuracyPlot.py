import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from pathlib import Path, PureWindowsPath

from numpy.distutils import pathccompiler
from scipy import stats
import random
from seaborn.external.husl import max_chroma

# –
from ViolinPlot import ViolinPlot


#
path = "C:/Users/Admin/Documents/HI/Machine Learning/Cod Otoliths/Predictions/" #/Python/Otholits/"#
names= [ "B6 minimum.txt", "B5 minimum.txt", "B4 minimum.txt" ,"B456 Ensamble min.txt","B6 middle.txt",
         "EfficientNetV2 m mid.txt","EfficientNetV2 L mid.txt","EfficientNetV2 L All.txt"]    # true value first

namev2= ["EfficientNetV2 m mid.txt","EfficientNetV2 L mid.txt", "EfficientNetV2 L All.txt"]

c= 2
color= (0.2,0.5,0.6- 0*0.1,0.4)
#colors=["powderblue","lightsteelblue","steelblue","navy","dodgerblue","lightskyblue"]
colors=[ "black", "red", "green", "blue", "steelblue", "Navy", "magenta","olive"]
#fname= path+name

def load_file(filename):
        fname= path+filename
        f = open(fname, 'r')
        xp = np.array([])
        xt = np.array([])
        for line in f:
            xt = np.append(xt, float(line.split()[0])) # true value first !
            xp = np.append(xp, float(line.split()[1])) # predicted value

        f.close()
        return xp,xt
# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

def defineAxis( title,axis_label):
        axisLabelSize = 20
        tickLabelSize = 18

        # plt.title(title+' age')
        #plt.ylabel('CNN Predicted ' + title + " age", fontsize=axisLabelSize)
        plt.xlabel( title + " age (years)", fontsize=axisLabelSize)
        plt.xticks(axis_label, fontsize=tickLabelSize)
        #plt.yticks(self.axis_label, fontsize=tickLabelSize)

#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

# Defines all the violin objects one for each name
def defineViolins(names):
    v= []
    index=0
    for name in names:
        v.append(ViolinPlot(name))
        violin= v[index]
        violin.prediction, violin.values = load_file(name)
        violin.maxa= int(violin.values.max())
        # – Create violin distribution
        violin.list2D()
        violin.findCountAndMax(False)
        violin.calculateAccuracy()
        violin.calculateMSE()
        index+=1
    return v
#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

#–– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
def accuracyList2D(violins,max_index=13):
    list=[]                                 # List elements are lists
    for age in range(max_index):
        list.append([])

    for age in range(max_index):
        for vi in violins:
            list[age].append(vi.accuracy[age])

    return list
#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

def MSEList2D(violins,max_index=13):
    list=[]                                 # List elements are lists
    for age in range(max_index):
        list.append([])

    for age in range(max_index):
        for vi in violins:
            list[age].append(vi.mse[age])

    return list
#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
# Scatter plot of all models
def scatterModel(isMSE=False):
    i=0
    ms=16
    col= "darkred"   #wheat cornsilk grey, silver darksalmon
    for vi in v:
        m= '*'
        w= 1.1
        c= colors[i]

        # if names[i].__contains__("All"):
        #     m= 'o'
        #     w=1.8
        #–

        if isMSE:
            plt.scatter(labels, vi.mse, marker=m, facecolors='none', edgecolors=col, alpha=0.5, s=ms)
        else :
            #plt.plot(labels,vi.accuracy,color=c,label=names, linestyle='solid',linewidth= w) #markersize=1)
            plt.scatter(labels, vi.accuracy, marker=m, facecolors='none', edgecolors='b', alpha=0.6, s=ms)
        i+=1
# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


# – The actual Violin plot
# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
def plotviolin(labels, list, count, isBox,co="black"):

    w= [0.05+float(i)/max(count) for i in count]
    w2= 0.1

    if isBox:
        plt.boxplot( list, notch=False, patch_artist=True, positions=labels,  widths=w2,showmeans=False)

        # Violin Plot
    else :
        parts=plt.violinplot( list, positions=labels, points=1000, widths=w,
                            showmeans=False, showextrema=False, showmedians=True,bw_method='silverman')
        # Colouring
        c = (0.2,0.5,0.6,0.2)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_edgecolor(c)
            pc.set_alpha(0.5)

#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


def generateText(max_index, models):
    mtext = "Models : "
    atext = "    Model Mean \n   Accuracy & MSE: "

    length= len(models)

    # Calculate accuracy and MSE ,average
    accuracy=[]
    MSE=[]
    count= models[0].count
    for age in range(max_index):
        a=0
        m=0
        for vi in models:
            a+=vi.accuracy[age]
            m+=vi.mse[age]
        a/= length
        m/= length
        accuracy.append(a)
        MSE.append(m)

    # print("mean accuracy ",accuracy)
    # print(" mean MSE ", MSE )

    for i in range(max_index):
        atext+= "\n"+ "{:03.0f}".format(i+1)+"  {:4.2f}". format(accuracy[i])
        atext+= "     {:4.2f}". format(MSE[i]) +"     n="+str(count[i])

         #"value {:.2f} %". format(100*v)

    print(atext)


    for x in names:
        mtext += "\n" + x.replace('.txt', '')

    return mtext,atext,accuracy
#– – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


name= names[0]

#– Create violin object
v=defineViolins(names)
violin=v[0]     #helper

maxage= int(violin.values.max())
print(f'{" max age: "} {maxage}')

#– Create labels
labels, axis_label= violin.create_labels()

# Find count
violin.findCountAndMax(False)
#print(violin.accuracy)

# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
# – Plotting –
# – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


#Set axis title, labels and annotation
ytics= [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ytics2=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]    # accuracy

defineAxis("Otolith",labels)
plt.ylabel("Accuracy %", fontsize=16)
plt.yticks(ytics2, fontsize=12) #MSE

# Convert to percentage %
for vi in v:
    vi.accuracy=  [element * 1 for element in vi.accuracy]

# List of accuracy for each model
listaccuracy=  accuracyList2D(v,violin.maxa)
listMSE= MSEList2D(v,violin.maxa)
#print("listcount: ",len(list))

# – Try MSE
plt.ylabel("MSE", fontsize=16)
plt.yticks(ytics, fontsize=12) #MSE
plt.ylim(0,1)
list= listMSE

# Violin or box
plotviolin(labels,list,violin.count, True)
#plotviolin(labels,listaccuracy,violin.count, True)
#scatterModel(True)
#scatterModel(False)


ax2 = plt.twinx()
ax2.set_ylabel('Accuracy', color="r")



mtext,atext,ac = generateText(maxage,v)
# accuracy line
plt.plot(labels,ac,color="r", linestyle='solid',linewidth= 1) #markersize=1)
plt.scatter(labels,ac, marker="o", facecolors='none', edgecolors='r', alpha=0.6, s=20)

# Display text about models
plt.text(x= .80, y= .60,s=mtext, fontsize= 14)
plt.text(x= .80, y= .2,s=atext, fontsize= 11)


plt.grid(axis='y', linestyle='dashed',linewidth= 0.3)
#plt.legend(names, fontsize=16)
plt.show()


#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
#colors = 'red'  # plt.cm.coolwarm(difference)    #blue= ["powderblue","lightsteelblue","steelblue","navy","dodgerblue","lightskyblue"]
