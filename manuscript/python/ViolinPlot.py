import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from pathlib import Path, PureWindowsPath

from numpy.distutils import pathccompiler
from scipy import stats
import random
from seaborn.external.husl import max_chroma

#
path = "C:/Users/Admin/Documents/HI/Machine Learning/Cod Otoliths/Predictions/" #/Python/Otholits/"#
names= [ "B6 minimum.txt", "B5 minimum.txt", "B4 minimum.txt" ,"B456 Ensamble min.txt","B6 middle.txt","b6 mid f0.txt" ]    # true value first
namev2= ["EfficientNetV2 m mid.txt","EfficientNetV2 L mid.txt", "EfficientNetV2 L All.txt"]

#name= "test predicted.txt"
c= 2
color= (0.2,0.5,0.6- 0*0.1,0.4)
#fname= path+name

#– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
class ViolinPlot:

    bb=1

    def __init__(self,title):
        #–  A list of lists
        self.violinList=[]
        self.violinListAc=[]
        #– Values = true values
        self.values= np.array([])
        self.acvalues= np.array([])
        #
        self.title=title
        self.axis_label=[]
        # Max age
        self.maxa=0

        #– Prediction values
        self.prediction= np.array([])
        self.acprediction= np.array([])
        #– Count is a list of #n predictions for each true value (class)
        self.count= []
        #– The maximum predicted value in each list (age class)
        self.maxv=[]
        #– MSE
        self.mse=[]
        #– Accuracy
        self.accuracy=[]
    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

    # calculatorAccuracy
    def calculateAccuracy(self):

        # Each bin of ages
        for i in range(self.maxa):
            age= int(i)+1
            ac=0
            a=0
            # Loop through predictions in each bin
            for x in range(self.count[i]):
                pred= self.violinList[i][x]
                d= pred - age
                a=  int(abs(round(pred) - age)==0)
                #print("age=",age, " prediction= ",pred, " accuracy = ",a)
                ac+= a
            # average
            ac/=self.count[i]
            self.accuracy.append(ac)


    # Mean Squared Error
    def calculateMSE(self):

        # Each bin of ages
        for i in range(self.maxa):
            age= int(i)+1
            # mean
            m=0
            # Loop through predictions in each bin
            for x in range(self.count[i]):
                pred= self.violinList[i][x]
                d= pred - age
                m+= np.square(d)
#                print(x," : age=",age, " diff=",d," m=",m)
            # average
            m/=self.count[i]
            self.mse.append(m)

        #print("MSE",self.mse)

    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    def accuratePredictions(self):
        index=0
        for age in self.values:
            pred= self.prediction[index]   # Get the corresponding predicted <-> true

            #– Extra list for accurate predictions
            if ( int(abs(round(pred) - age)==0) ):
                #print(index, " age:",age ," prediction->",pred)
                self.acprediction=np.append( self.acprediction,pred)#.append(pred)
                self.acvalues= np.append( self.acvalues,age) #self.acvalues.append(age)
            index+=1



    def addListViolin(self, moreValues, max_index=13):

        # This index takes care of the mapping between predicted and true values
        index=0
        for age in self.values:
            ix= int(age)-1                   # Index is informed by list_t -> 0 corresponding to 1-year-olds et cetera…
            pred=   moreValues[index] #self.prediction[index]   # Get the corresponding number:  predicted <-> true
            if ix<max_index:
                # This is where we add the predicted number to the correct list(1 of max age) informed by list_t !
                self.violinList[ix].append(pred)
                #– Extra list for accurate predictions
                # if ( int(abs(round(pred) - age)==0) ):
                #     self.violinListAc[ix].append(pred)
            index+=1

        #self.findCountAndMax(max_index,False)
        # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –



    #–– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

    #   Creates a list of lists – necessary for "violin" plot
    #–– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    def list2D(self, max_index=13):

        # List elements are lists
        for age in range(max_index):
            self.violinList.append([])
            self.violinListAc.append([])    #extra list for accurate predictions


        # This index takes care of the mapping between predicted and true values
        index=0
        for age in self.values:
            ix= int(age)-1                   # Index is informed by list_t -> 0 corresponding to 1-year-olds et cetera…
            pred= self.prediction[index]   # Get the corresponding number:  predicted <-> true
            if ix<max_index:
                # This is where we add the predicted number to the correct list(1 of max age) informed by list_t !
                self.violinList[ix].append(pred)
                #– Extra list for accurate predictions
                if ( int(abs(round(pred) - age)==0) ):
                    self.violinListAc[ix].append(pred)
            index+=1

        #self.findCountAndMax(max_index,False)
        # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – –
    # Collect the count and maximum for each class(age)
    # The count is the number of predictions for each class(age)
    # Maxv is the Max value for each count list
    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – –
    def findCountAndMax(self, doPrint=True):
        self.count=[]

        for age in range(self.maxa):
            self.count.append(len(self.violinList[age]))
            self.maxv.append(max(self.violinList[age]))
            if doPrint:
                print(" List content: " + str(age) + " Entries " + str(self.count[age]))
                print(self.violinList[age])
                print(" maximum: " + str(max(self.violinList[age])))

    # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –

    def create_labels(self):
        labels = []
        axis_label = []
        for x in range(self.maxa):
            labels.append(x + 1)
            axis_label.append(x)
        #axis_label.append(max_value)
        return labels,axis_label
    # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


    # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    # – The actual Violin plot
    # – – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    def plot(self,labels):
        fig, axis = plt.subplots(nrows=1, ncols=1)

        w= [0.05+float(i)/max(self.count) for i in self.count]

        print(w)
        # Axis
        axis.yaxis.grid(True, color = (0.9, 0.9, 0.9), linewidth=0.5, linestyle='-') # "lightgrey")
        axis.set_axisbelow(True)
        # Violin Plot
        parts=axis.violinplot( self.violinList, positions=labels, points=1000, widths=w,
                    showmeans=False, showextrema=False, showmedians=False,bw_method='silverman')
        # monkey

        # Colouring
        c = color   #(0.2,0.5,0.6,0.4)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_edgecolor(c)
            pc.set_alpha(0.5)
    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –


    #– – – – – – – – – – – – – – – – – – – –
    # Plot  Scatter
    #– – – – – – – – – – – – – – – – – – – –
    def plotScatter(self,v,p, alpha=0.15, size=20):
        colors = 'navy'  # plt.cm.coolwarm(difference)    #blue= ["powderblue","lightsteelblue","steelblue","navy","dodgerblue","lightskyblue"]
        plt.scatter(v, p, marker='o', c=colors, alpha=alpha, s=size)

        maxrange = self.maxa + 0.2
        plt.xlim(0.5, maxrange)
        plt.ylim(0, maxrange + 0.5)

    def plotScatterAccuracy(self,alpha=0.15, size= 15):
        colors = 'red'  # plt.cm.coolwarm(difference)    #blue= ["powderblue","lightsteelblue","steelblue","navy","dodgerblue","lightskyblue"]
        plt.scatter(self.acvalues+0.05, self.acprediction, marker='o', c=colors, alpha=alpha, s=size)
        maxrange = self.maxa+ 0.2
        plt.xlim(0.5, maxrange)

        plt.ylim(0, maxrange + 0.5)


    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    def defineAxis(self, title):
        axisLabelSize = 24
        tickLabelSize = 20

        # plt.title(title+' age')
        plt.ylabel('CNN Predicted ' + title + " age", fontsize=axisLabelSize)
        plt.xlabel('Labelled ' + title + " age (years)", fontsize=axisLabelSize)
        plt.xticks(self.axis_label, fontsize=tickLabelSize)
        plt.yticks(self.axis_label, fontsize=tickLabelSize)

    #– – – – – – – – – – – – – – – – – – – –  – – – – – – – – – – – – – – – – – – – –
    def annotatedCount(self,size=15):
        # – fontsize for annotated numbers
        plt.rcParams.update({'font.size': size})
        # – Text annotation above each – #               #elephant
        for i in range(0, len(self.maxv)):
            plt.rcParams.update({'font.size': size})
            plt.text(labels[i], 0.2 + self.maxv[i], violin.count[i])
            # – MSE
            plt.rcParams.update({'font.size': size-3-0.1*i})
            text= "Acc {:.2f} %". format(100*violin.accuracy[i])
            plt.text(labels[i]-0.35, 0.9 + self.maxv[i], text)
        #
        plt.text( 1 , 10 ,self.title)
#– – – – – – – – – – – – – – – – – – – – CLASS definition end – – – – – – – – – – – – – – – – – – – –


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




if __name__ == '__main__':

    # – Read values from text file
    c=3
    name= names[c]
    #name= namev2[c]

    # – Create violin object
    violin = ViolinPlot(name)

    violin.prediction, violin.values = load_file(name)
    #violin.accuratePredictions()
    #
    violin.maxa= int(violin.values.max())
    print(f'{" max age: "} {violin.maxa}')

    # – Create violin distribution
    violin.list2D()
    # ~

    # Add values
    p,v = load_file(names[1])
    p3,v3 = load_file(names[2])
    #violin.addListViolin(p)
    #violin.addListViolin(p3)
    violin.title= name
    #violin.title+=  "\n"+names[1]+"\n"+names[2]
    # Find count
    violin.findCountAndMax(False)
    print('{"count: "}', violin.count)
    #print(violin.count)


    # – Create labels
    labels, violin.axis_label= violin.create_labels()
    violin.axis_label.append(violin.maxa)    # hacking
    violin.calculateMSE()
    violin.calculateAccuracy()

    # – The Violin Plot
    violin.plot(labels)
    # Add scatter, one point for each prediction
    violin.plotScatter(violin.values, violin.prediction)
#    violin.plotScatter(v, p)
#    violin.plotScatter(v3, p3)
    violin.plotScatterAccuracy()

    # Add x=y line plot
    plt.plot(violin.axis_label,violin.axis_label,'b',label='xy', linestyle='dashed',linewidth= 0.4) #markersize=1)
    #plt.plot(t,p,'b',label='xy', linestyle='dashed',linewidth= 0.4) #markersize=1)

    # Accuracy line
    #plt.plot(labels,violin.accuracy,'r',label='Acc', linestyle='dashed',linewidth= 0.4) #markersize=1)

    # Set axis title, labels and annotation
    violin.defineAxis("Otoliths")
    violin.annotatedCount()

    plt.show()


"""
#difference= calculateDifference(listpredicted,listtrue)
 # for i,x in enumerate(t):
    #     print(f'{i}{" : "} {x}{" --> "}{p[i]}')

# f= open(path+name,'r')    #.read().splitlines()
# lines=f.readlines()



        # zip_object = zip(list1, list2)
        # d=[]
        # for list1_i, list2_i in zip_object:
        #     d.append(list1_i-list2_i)

"""
