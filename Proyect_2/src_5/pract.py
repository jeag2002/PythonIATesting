'''
PRAC-1
'''

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
import os
import numpy as np
import random, sys, math, operator
import matplotlib.pyplot as plot
import pylab as pl
import itertools



#VARIABLES GLOBALES
###############################################################################

path_to_data_feat_train = ".././data_5/X_train.txt";
path_to_data_feat_type_train = ".././data_5/y_train.txt";
path_to_data_feat_test = ".././data_5/X_test.txt";
path_to_data_feat_type_test = ".././data_5/y_test.txt";

data_feat_train = [];
data_info_train = [];

data_feat_test =  [];
data_info_test = [];

file_train = 0;
file_test = 0;
###############################################################################


print "EJERCICIO -1 : Carga y preprocesado de datos"
print "#####################################################################################################"

#train-data
for line in open( path_to_data_feat_train,"r"):
    content = line.split();
    data = [];
    for x in range(len(content)):
        data.append(float(content[x]));
    
    offset = 561 - len(data);
    if (offset > 0):
        print "line " + str(file_train) +  " offset " + str(offset)
        for x in range(offset):
            data.append(float(1.0));   
    
    data_feat_train.append(data);
    file_train += 1;
    
#train-info    
for line in open( path_to_data_feat_type_train,"r"):
    content = line.split();
    data = [];
    for x in range(len(content)):
        data_info_train.append(float(content[x]));
        
#test-data        
for line in open( path_to_data_feat_test,"r"):
    content = line.split();
    data = [];
    for x in range(len(content)):
        data.append(float(content[x]));
        
    offset = 561 - len(data);
    if (offset > 0):
        print "line " + str(file_train) +  " offset " + str(offset)
        for x in range(offset):
            data.append(float(1.0));    
        
    data_feat_test.append(data);
    file_test += 1;

#test-info
for line in open( path_to_data_feat_type_train,"r"):
    content = line.split();
    data = [];
    for x in range(len(content)):
        data_info_test.append(float(content[x]));
        
print "#####################################################################################################"

print "EJERCICIO -2 : Discriminacion variables por PCA"
print "#####################################################################################################"

dx = np.array(data_feat_train).reshape(len(data_feat_train),561);

prom = list(map(np.average, dx))
stds = list(map(np.std, dx))
    
values = [list(map(lambda x: (x - prom[i]) / stds[i], dx[i]))
             for i in range(len(dx))]


dx_1 = np.array(data_feat_test).reshape(len(data_feat_test),561);

prom = list(map(np.average, dx_1))
stds = list(map(np.std, dx_1))
    
values_1 = [list(map(lambda x: (x - prom[i]) / stds[i], dx_1[i]))
             for i in range(len(dx_1))]

#561 Componentes (TRAIN)
#######################################################################
mypca = PCA(n_components=561)
mypca.fit(values)
values_proj = mypca.transform(values)

print "TRAIN 561 componentes filas " + str(len(values_proj)) + " columnas " + str(len(values_proj[0]))


fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values_proj[:,0],values_proj[:,1],marker='o',c='blue')
sp.scatter(values_proj[:,0],values_proj[:,2],marker='o',c='red')
sp.scatter(values_proj[:,0],values_proj[:,3],marker='o',c='green')
sp.scatter(values_proj[:,0],values_proj[:,4],marker='o',c='yellow')
sp.scatter(values_proj[:,0],values_proj[:,5],marker='o',c='magenta')
sp.scatter(values_proj[:,0],values_proj[:,6],marker='o',c='cyan')
sp.scatter(values_proj[:,0],values_proj[:,7],marker='o',c='black')
sp.scatter(values_proj[:,0],values_proj[:,8],marker='o',c='pink')
sp.scatter(values_proj[:,0],values_proj[:,9],marker='o',c='gray')
plot.show()



#######################################################################

#561 Componentes (TEST)
#######################################################################
mypca_T = PCA(n_components=561)
mypca_T.fit(values_1)
values_proj_T = mypca_T.transform(values_1)

print "TEST 561 componentes filas " + str(len(values_proj_T)) + " columnas " + str(len(values_proj_T[0]))

fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values_proj_T[:,0],values_proj_T[:,1],marker='o',c='blue')
sp.scatter(values_proj_T[:,0],values_proj_T[:,2],marker='o',c='red')
sp.scatter(values_proj_T[:,0],values_proj_T[:,3],marker='o',c='green')
sp.scatter(values_proj_T[:,0],values_proj_T[:,4],marker='o',c='yellow')
sp.scatter(values_proj_T[:,0],values_proj_T[:,5],marker='o',c='magenta')
sp.scatter(values_proj_T[:,0],values_proj_T[:,6],marker='o',c='cyan')
sp.scatter(values_proj_T[:,0],values_proj_T[:,7],marker='o',c='black')
sp.scatter(values_proj_T[:,0],values_proj_T[:,8],marker='o',c='pink')
sp.scatter(values_proj_T[:,0],values_proj_T[:,9],marker='o',c='gray')
plot.show()


#######################################################################


#10 Componentes (TRAIN)
#######################################################################
mypca_1 = PCA(n_components=10)
mypca_1.fit(values)
values_proj_1 = mypca_1.transform(values)

print "TRAIN 10 componentes filas " + str(len(values_proj_1)) + " columnas " + str(len(values_proj_1[0]))   
   
tags = [[i]*200 for i in range(10)]
tags = list(itertools.chain(*tags))

fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values_proj_1[:,0],values_proj_1[:,1],marker='o',c='blue')
sp.scatter(values_proj_1[:,0],values_proj_1[:,2],marker='o',c='red')
sp.scatter(values_proj_1[:,0],values_proj_1[:,3],marker='o',c='green')
sp.scatter(values_proj_1[:,0],values_proj_1[:,4],marker='o',c='yellow')
sp.scatter(values_proj_1[:,0],values_proj_1[:,5],marker='o',c='magenta')
sp.scatter(values_proj_1[:,0],values_proj_1[:,6],marker='o',c='cyan')
sp.scatter(values_proj_1[:,0],values_proj_1[:,7],marker='o',c='black')
sp.scatter(values_proj_1[:,0],values_proj_1[:,8],marker='o',c='pink')
sp.scatter(values_proj_1[:,0],values_proj_1[:,9],marker='o',c='gray')
plot.show()
#######################################################################


#10 Componentes (TEST)
#######################################################################
mypca_1T = PCA(n_components=10)
mypca_1T.fit(values_1)
values_proj_1T = mypca_1.transform(values_1)

print "TEST 10 componentes filas " + str(len(values_proj_1T)) + " columnas " + str(len(values_proj_1T[0]))  


tags = [[i]*200 for i in range(10)]
tags = list(itertools.chain(*tags))
    
fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,1],marker='o',c='blue')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,2],marker='o',c='red')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,3],marker='o',c='green')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,4],marker='o',c='yellow')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,5],marker='o',c='magenta')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,6],marker='o',c='cyan')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,7],marker='o',c='black')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,8],marker='o',c='pink')
sp.scatter(values_proj_1T[:,0],values_proj_1T[:,9],marker='o',c='gray')
plot.show()

#######################################################################


fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values[:][0],values[:][1],marker='o',c='blue')
sp.scatter(values[:][0],values[:][2],marker='o',c='red')
sp.scatter(values[:][0],values[:][3],marker='o',c='green')
sp.scatter(values[:][0],values[:][4],marker='o',c='yellow')
sp.scatter(values[:][0],values[:][5],marker='o',c='magenta')
sp.scatter(values[:][0],values[:][6],marker='o',c='cyan')
sp.scatter(values[:][0],values[:][7],marker='o',c='black')
sp.scatter(values[:][0],values[:][8],marker='o',c='pink')
sp.scatter(values[:][0],values[:][9],marker='o',c='gray')
plot.show()


fig1 = plot.figure()
sp = fig1.gca()       
sp.scatter(values_1[:][0],values_1[:][1],marker='o',c='blue')
sp.scatter(values_1[:][0],values_1[:][2],marker='o',c='red')
sp.scatter(values_1[:][0],values_1[:][3],marker='o',c='green')
sp.scatter(values_1[:][0],values_1[:][4],marker='o',c='yellow')
sp.scatter(values_1[:][0],values_1[:][5],marker='o',c='magenta')
sp.scatter(values_1[:][0],values_1[:][6],marker='o',c='cyan')
sp.scatter(values_1[:][0],values_1[:][7],marker='o',c='black')
sp.scatter(values_1[:][0],values_1[:][8],marker='o',c='pink')
sp.scatter(values_1[:][0],values_1[:][9],marker='o',c='gray')
plot.show()




print "#####################################################################################################"

print "EJERCICIO -3: "
print "#####################################################################################################"


#############################
clf = GaussianNB()
clf.fit(values_proj, data_info_train)

accuracy = 0;

for i in range(len(values_proj_T)):
    data = [];
    data = values_proj_T[i];
    x = clf.predict(data);
    if (x == data_info_test[i]):
        accuracy += 1
        
accuracy = float(accuracy)/float(len(data_info_test));

print "accuracy con GaussianNB (con 561 componentes) " + str(accuracy) 


#############################
clf_1 = GaussianNB()
clf_1.fit(values_proj_1, data_info_train)

accuracy = 0;

for i in range(len(values_proj_1T)):
    data = [];
    data = values_proj_1T [i];
    x = clf_1.predict(data);
    if (x == data_info_test[i]):
        accuracy += 1
        
accuracy = float(accuracy)/float(len(data_info_test));

print "accuracy con GaussianNB (con 10 componentes) " + str(accuracy) 

#############################
clf_2 = LDA()
clf_2.fit(values_proj, data_info_train)

accuracy = 0;

for i in range(len(values_proj_T)):
    data = [];
    data = values_proj_T[i];
    x = clf_2.predict(data);
    if (x == data_info_test[i]):
        accuracy += 1
        
accuracy = float(accuracy)/float(len(data_info_test));

print "accuracy con LDA (con 561 componentes) " + str(accuracy) 

#############################
clf_3 = LDA()
clf_3.fit(values_proj_1, data_info_train)

accuracy = 0;

for i in range(len(values_proj_1T)):
    data = [];
    data = values_proj_1T[i];
    x = clf_3.predict(data);
    if (x == data_info_test[i]):
        accuracy += 1
        
accuracy = float(accuracy)/float(len(data_info_test));

print "accuracy con LDA (con 10 componentes) " + str(accuracy) 


print "#####################################################################################################"


