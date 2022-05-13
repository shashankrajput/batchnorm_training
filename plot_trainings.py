from os import listdir
from os.path import isfile, join

import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import re

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8',  '#4daf4a','#ff7f00',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']) 

matplotlib.rc('lines', linewidth=2)


path='./results_new/ResNet34/'
files = [join(path, f) for f in listdir(path)]


x_list=[]
accuracyDict={}

# f=join(path,'LTH_cifar10_resnet20.csv')
# with open(f) as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for (i, row) in enumerate(reader):
#         if i==0:
#             continue
#         if i==1:
#             accuracyDict['scratch']=[float(row[2])]
#             x_list.append(0)
        
#         if float(row[1])<100:
#             break
#         accuracyDict['scratch'].append(float(row[0]))
#         x_list.append(i)

# if len(accuracyDict['scratch'])!=len(x_list):
#     print("Error!!!!")

x_list=[i for i in range(100)]
for f in files:
    with open(f) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for (i, row) in enumerate(reader):
            
            accuracy=float(row[2])
            keystr=f[41:-4]
            if keystr not in accuracyDict:
                accuracyDict[keystr]=[]

            accuracyDict[keystr].append(float(row[2]))
        
        # if len(accuracyDict[keystr])!=len(x_list):
        #     print(len(accuracyDict[keystr]))
        #     print(len(accuracyDict[x_list]))
        #     print("Error!!!!")


for keystr, accuracy_list in accuracyDict.items():
    if 'False' in keystr:
        plt.plot(x_list, accuracy_list, linestyle='dashed', label='Train all')
    else:
        plt.plot(x_list, accuracy_list, linestyle='solid', label='Batch norm only, sparsity='+keystr[28:])
    # except:
    #     print(keystr, accuracy_list)
    # if sparsity==100:
    #     plt.plot(x_list, accuracy_list, linestyle='dashed', label=str('scratch'))
    # else: 
    #     plt.plot(x_list, accuracy_list, linestyle='solid', label="sparsity="+str(sparsity)+", quantization="+str(quant))

handles, labels = plt.gca().get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))


plt.legend(handles,labels,  loc='right',fontsize=8,ncol=1, title='Accuracy of training only batch norm', bbox_to_anchor=(1.60, 0.5))
plt.xlabel(r'Number of epochs $K$',fontsize=12)
plt.ylabel(r'Test Accuracy',fontsize=12)

# ax1 = plt.gca()
# ax1.set_xticks([30, 40, 60, 90, 150, 220, 300])
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.savefig('Training.pdf',bbox_inches="tight")  

