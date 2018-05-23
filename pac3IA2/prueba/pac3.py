'''
'PAC 3


URLS CONSULTADAS

1)ADABOOST
https://engineering.purdue.edu/kak/Tutorials/AdaBoost.pdf
http://www.csd.uwo.ca/~olga/Courses//Fall2005/Lecture4-2.pdf
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

2)SVM-SVC
#http://scikit-learn.org/stable/modules/svm.html
#http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#example-svm-plot-svm-regression-py
#http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#example-svm-plot-custom-kernel-py

3)KERNEL CITYBLOCK
#http://nullege.com/codes/search/scipy.ndimage.distance_transform_cdt
#http://structure.usc.edu/numarray/node91.html
'''

import math;
from itertools import repeat;
from functools import reduce;

from adaboost import adaBoost;
from decisionTrees import DTs;

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pylab as pl
import numpy as np

#import scipy.ndimage as ndimage


#Variables globales
###################################################################################
data_Bankruptcy_BayesTrain = [];
data_Bankruptcy_BayesData = [];
data_Bankruptcy_Total = [];
data_Bankruptcy_Tot_ADA = []

data_Bankruptcy_Total_SVM_N = []; #equivalencia numerica
data_Bankruptcy_Total_ADA = [];
validacion_cruzada = 15; # k tamano del buck a probar

ADA_Target = [];
ADA_Values = [];
###################################################################################


#implementacion UOC libro NaiveBayes
#Aplicar solucion de Gerard Escudero Bakx. Implementar contar2(m[i],m[0]) en vez de contar2(m[1],m[0])
#Aplicar solucion de Silvina Re Gomez. Poner todas las divisiones a float. Si no no funciona. 
#Aplicar solucion de Pere Pares Casellas set pop(0) en vez de pop(6) en todas las implementaciones (ADABOOST, NaiveBayes, DTs)
####################################################################################

def contar(l):
    p = {}
    for x in l:
        p.setdefault(x,0);
        p[x]+=1
    return p

def contar2(l,k):
    p = {}
    for i in range(len(l)):
        p.setdefault(k[i],{})
        p[k[i]].setdefault(l[i],0)
        p[k[i]][l[i]] +=1
    return p

def Pxik(atr,N,k,Nk,n):
    if atr in N[k]:
        return float(N[k][atr])/float(Nk[k])
    else:
        return float(Nk[k])/float(n ** 2)

def classify(t,Nk,Nxik,n):
    
    numero_t = len(t);
    l = [(k, Nk[k]/n*reduce(lambda x,y : x*y, map(Pxik,t,Nxik,[k for a in range(len(t))],[Nk]*numero_t,[n]*numero_t))) for k in Nk.keys()]
    
    return max(l, key=lambda x: x[1])[0]

        
def naiveBayesBookUOC(test, train):
    
    n = len(train);
    numero_test = len(test);
    m = list(zip(*train))
    Nk = contar(m[0])
    Nxik = [contar2(m[i],m[0]) for i in range(1, len(m))]
    classes = list(map(lambda x: x.pop(0), test))
    prediccions = list(map(classify,test,[Nk]*numero_test,[Nxik]*numero_test,[n]*numero_test))
    res = (float(len(list(filter(lambda x: x[0] == x[1], zip(*[prediccions,classes])))))/float(len(test)))
    return res;
####################################################################################

path_Qualitative_Bankruptcy = ".././data/Qualitative_Bankruptcy.data.txt";
data = [];
incdata = 0;

def data_number_equivalency(data):
    if data == 'NB':
        return 0
    elif data == 'B':
        return 1
    elif data == 'A':
        return 2
    elif data == 'P':
        return 3
    elif data == 'N':
        return 4
    else:
        return 5
      
for line in open( path_Qualitative_Bankruptcy,"r"):
    content = line.split(',');
    classe = content[6][:-1];
    
    data.append(classe);
    data.append(content[0]);
    data.append(content[1]);
    data.append(content[2]);
    data.append(content[3]);
    data.append(content[4]);
    data.append(content[5]);
    data_Bankruptcy_Total.append(data);
    data = [];
    
    data.append(data_number_equivalency(classe));
    data.append(data_number_equivalency(content[0]));
    data.append(data_number_equivalency(content[1]));
    data.append(data_number_equivalency(content[2]));
    data.append(data_number_equivalency(content[3]));
    data.append(data_number_equivalency(content[4]));
    data.append(data_number_equivalency(content[5]));
    data_Bankruptcy_Total_SVM_N.append(data);
    data = [];


print "EJERCICIO 1 Estadisticas asignando los datos de entrenamiento los mismos que los datos de validacion"
print "========================================================================================"
accuracy = naiveBayesBookUOC(data_Bankruptcy_Total, data_Bankruptcy_Total);
print "accuracy - NaiveBayes TOTAL " + str(accuracy);
               
accuracy = adaBoost(data_Bankruptcy_Total, data_Bankruptcy_Total,False);
print "accuracy - ADABOOST TOTAL " + str(accuracy);

[tree, accuracy] = DTs(data_Bankruptcy_Total, data_Bankruptcy_Total);
print "accuracy - DTs TOTAL " + str(accuracy);
print "========================================================================================"


data_Bankruptcy_BayesTrain = []
data_Bankruptcy_BayesData = []
#data_Bankruptcy_Total = []

print "EJERCICIO 1 Estadisticas escogiendo un elemento de test cada  K = " + str(validacion_cruzada) + " elementos"
print "========================================================================================"

incr = 0

for line in open( path_Qualitative_Bankruptcy,"r"):
    content = line.split(',');
    classe = content[6][:-1];
    data.append(classe);
    data.append(content[0]);
    data.append(content[1]);
    data.append(content[2]);
    data.append(content[3]);
    data.append(content[4]);
    data.append(content[5]);
    if (incr < validacion_cruzada):
        data_Bankruptcy_BayesTrain.append(data);
        incr += 1;
    elif (incr == validacion_cruzada):
        data_Bankruptcy_BayesData.append(data);
        incr = 0;
    else:
        incr = 0;
    data = [];


accuracy = naiveBayesBookUOC(data_Bankruptcy_BayesData, data_Bankruptcy_BayesTrain);
print "accuracy - NAIVEBAYES TEST-K " + str(accuracy);
               
accuracy = adaBoost(data_Bankruptcy_BayesTrain, data_Bankruptcy_BayesData,False);
print "accuracy - ADABOOST TEST-K " + str(accuracy);

[tree, accuracy] = DTs(data_Bankruptcy_BayesTrain, data_Bankruptcy_BayesTrain);
print "accuracy - DECISIONTREE TEST-K " + str(accuracy);

print "========================================================================================"

accuracy = 0
index = 0
buck = len(data_Bankruptcy_Total) / validacion_cruzada;
nBackruptcyTotal = len(data_Bankruptcy_Total)

data_Bankruptcy_BayesTrain = []
data_Bankruptcy_BayesData = []
data_Bankruptcy_Total = []

#aplicamos validacion cruzada con K=10 y evaluamos la precision de los 
#diferentes motores de clasificacion

print "EJERCICIO 1 Estadisticas realizando validacion cruzada con K = " + str(validacion_cruzada) + " tamano del vector de datos " + str(buck) + " tamano del vector de entrenamiento " + str(175-buck);
print "========================================================================================"

while  (index < nBackruptcyTotal):
    
    for line in open( path_Qualitative_Bankruptcy,"r"):
        content = line.split(',');
        classe = content[6][:-1];
        data.append(classe);
        data.append(content[0]);
        data.append(content[1]);
        data.append(content[2]);
        data.append(content[3]);
        data.append(content[4]);
        data.append(content[5]);
        data_Bankruptcy_Total.append(data);
        data = [];
    
    buck_aux = 0;
    
    #set Data
    if (index + buck < len(data_Bankruptcy_Total)):
        limit = index+buck
        x=index; 
        while (x<limit):
            data_Bankruptcy_BayesData.append(data_Bankruptcy_Total[x])
            x+=1;
    else:
        limit = len(data_Bankruptcy_Total) - index;
        x=index;
        while (x<limit):
            data_Bankruptcy_BayesData.append(data_Bankruptcy_Total[x]);
            x+=1;
    
    #set train
    x=0; 
    while (x<index):
        data_Bankruptcy_BayesTrain.append(data_Bankruptcy_Total[x]);
        x+=1;
    
    x = index+buck;
    while (x<len(data_Bankruptcy_Total)):
        data_Bankruptcy_BayesTrain.append(data_Bankruptcy_Total[x]);
        x+=1;
    
    
    if (len(data_Bankruptcy_BayesData) > 0):
        #set de valores para los motores adaboost de sklearn
        
        ADA_Target = [];
        ADA_Values = [];
        
        
        for i in range(len(data_Bankruptcy_BayesTrain)):
            data = data_Bankruptcy_BayesTrain[i]
            data_value = [];
            ADA_Target.append(data_number_equivalency(data[0]));
            data_value.append(data_number_equivalency(data[1]));
            data_value.append(data_number_equivalency(data[2]));
            data_value.append(data_number_equivalency(data[3]));
            data_value.append(data_number_equivalency(data[4]));
            data_value.append(data_number_equivalency(data[5]));
            data_value.append(data_number_equivalency(data[6]));
            ADA_Values.append(data_value);
            data_value = [];
            data = [];
        
        accuracy = naiveBayesBookUOC(data_Bankruptcy_BayesTrain, data_Bankruptcy_BayesData);
        print "accuracy buck " + str(index) + " - NAIVEBAYES " + str(accuracy);
            
        #Tal y como dicen en los foros de la asignatura, siempre me da 100% para todas las iteraciones    
        accuracy = adaBoost(data_Bankruptcy_BayesTrain, data_Bankruptcy_BayesData,False);
        print "accuracy buck " + str(index) + " - ADABOOST " + str(accuracy);
        
         
        clf = AdaBoostClassifier(n_estimators=100);
        scores = cross_val_score(clf, np.array(ADA_Values),np.array(ADA_Target));
        accuracy = scores.mean();
        print "accuracy buck " + str(index) + " - ADABOOSTCLASSIFIER - sklearn " + str(accuracy);   
            
        ada_discrete = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1),
                learning_rate=1,
                n_estimators=100,
                algorithm="SAMME")
        obj = ada_discrete.fit(ADA_Values, ADA_Target)
        accuracy = ada_discrete.score(ADA_Values, ADA_Target)
            
        print "accuracy buck " + str(index) + " - ADABOOSTCLASSIFIERDECISIONTREEDISCRETE - sklearn " + str(accuracy);
            
        ada_real = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1),
                learning_rate=1,
                n_estimators=100,
                algorithm="SAMME.R")
        obj = ada_real.fit(ADA_Values, ADA_Target)
        accuracy = ada_real.score(ADA_Values, ADA_Target);
        print "accuracy buck " + str(index) + " - ADABOOSTCLASSIFIERDECISIONTREEREAL - sklearn " + str(accuracy);
            
        [tree, accuracy] = DTs(data_Bankruptcy_BayesTrain, data_Bankruptcy_BayesTrain);
        print "accuracy buck " + str(index) + " - DECISIONTREE " + str(accuracy) +  " - tree " + str(tree);
        
        
        dt_real = DecisionTreeClassifier(random_state=0)
        obj = dt_real.fit(ADA_Values, ADA_Target);
        accuracy = ada_real.score(ADA_Values, ADA_Target);
        print "accuracy buck " + str(index) + " - DECISIONTREECLASSIFIER - sklearn " + str(accuracy);
        
        
        

    index+=buck;
    
    data_Bankruptcy_BayesTrain = []
    data_Bankruptcy_BayesData = []
    data_Bankruptcy_Total = []

print "========================================================================================"    


print "EJERCICIO 2 - precomputed KERNEL SVM"
print "========================================================================================"    


#KERNEL EUCLIDEAN (se hace como prueba, transformando previamente los datos categoricos en datos numericos)
########################################################


def ownKernelEuclidean(X,Y):
    
    data_X = X;
    data_Y = Y.T; 
    
    element = 0;
    Z = [[0 for j in range(len(data_Y[0]))] for i in range(len(data_X))];
    for i in range(len(data_X)):
        for j in range(len(data_Y[0])):  
            for k in range(len(data_X[0])):
                element += math.pow((data_X[i][k] - data_Y[k][j]),2)  
            Z[i][j] = math.sqrt(element);
            element = 0
    return Z;

########################################################


#KERNEL HAMMING (en principio es el unico valido para las caracteristicas que se requieren del programa)
########################################################
def sigma(a,b):
    if (a==b):
        return 0;
    else:
        return 1;

def ownKernelHamming(X,Y):
    
    data_X = X;   #158x6
    data_Y = Y.T; #6x158
    
    element = 0;
    Z = [[0 for j in range(len(data_Y[0]))] for i in range(len(data_X))];
    for i in range(len(data_X)):
        for j in range(len(data_Y[0])):  
            for k in range(len(data_X[0])):
                element += sigma(data_X[i][k],data_Y[k][j]) 
            element = 6-element;  
            Z[i][j] = element;
            element = 0
    return Z;

########################################################

#no se implementa el kernel "City Block:" 
#evalua categorias como binarios. (serviria para evaluar diferentes "targets" pero no "values")

#no se implementa el kernel "Stringlike Features:"
#evalua strings

data_Bankruptcy_BayesTrain = []
data_Bankruptcy_BayesData = []
data_Bankruptcy_Total = []

accuracy = 0
index = 0

while  (index < nBackruptcyTotal):
    
    for line in open( path_Qualitative_Bankruptcy,"r"):
        content = line.split(',');
        classe = content[6][:-1];
        data.append(classe);
        data.append(content[0]);
        data.append(content[1]);
        data.append(content[2]);
        data.append(content[3]);
        data.append(content[4]);
        data.append(content[5]);
        data_Bankruptcy_Total.append(data);
        data = [];


    #set Data
    if (index + buck < len(data_Bankruptcy_Total)):
        limit = index+buck
        x=index; 
        while (x<limit):
            data_Bankruptcy_BayesData.append(data_Bankruptcy_Total[x])
            x+=1;
    else:
        limit = len(data_Bankruptcy_Total) - index;
        x=index;
        while (x<limit):
            data_Bankruptcy_BayesData.append(data_Bankruptcy_Total[x]);
            x+=1;
    
    #set train
    x=0; 
    while (x<index):
        data_Bankruptcy_BayesTrain.append(data_Bankruptcy_Total[x]);
        x+=1;
    
    x = index+buck;
    while (x<len(data_Bankruptcy_Total)):
        data_Bankruptcy_BayesTrain.append(data_Bankruptcy_Total[x]);
        x+=1;
    
    
    if (len(data_Bankruptcy_BayesData) > 0):
        #set de valores para los motores SVM-SVC de sklearn
        
        #valores de entrenamiento
        ADA_Target_Train = []; #identifica las clases (B,NB)
        ADA_Values_Train = []; #identifica los valores  (P,A,N)
        
        #valores de testing
        ADA_Target_Data = []; #identifica las clases (B,NB)
        ADA_Values_Data = []; #identifica los valores (P,A,N)
        
        
        
        for i in range(len(data_Bankruptcy_BayesTrain)):
            data = data_Bankruptcy_BayesTrain[i]
            data_value = [];
            ADA_Target_Train.append(data_number_equivalency(data[0]));
            data_value.append(data_number_equivalency(data[1]));
            data_value.append(data_number_equivalency(data[2]));
            data_value.append(data_number_equivalency(data[3]));
            data_value.append(data_number_equivalency(data[4]));
            data_value.append(data_number_equivalency(data[5]));
            data_value.append(data_number_equivalency(data[6]));
            ADA_Values_Train.append(data_value);
            data_value = [];
            data = [];
        
        for i in range(len(data_Bankruptcy_BayesData)):
            data = data_Bankruptcy_BayesData[i]
            data_value = [];
            ADA_Target_Data.append(data_number_equivalency(data[0]));
            data_value.append(data_number_equivalency(data[1]));
            data_value.append(data_number_equivalency(data[2]));
            data_value.append(data_number_equivalency(data[3]));
            data_value.append(data_number_equivalency(data[4]));
            data_value.append(data_number_equivalency(data[5]));
            data_value.append(data_number_equivalency(data[6]));
            ADA_Values_Data.append(data_value);
            data_value = [];
            data = [];
            
        
        clf = svm.SVC(kernel='linear'); 
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
        
        print "accuracy buck " + str(index) + " svm SVC LINEAL " +  str(accuracy)  
        
        clf = svm.SVC(kernel='rbf', C=1e3, gamma=0.1);
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
        
        
        print "accuracy buck " + str(index) + " svm SVC GAUSSIAN " +  str(accuracy) 
        
        clf = svm.SVC(kernel='poly', C=1e3, degree=4, coef0=1); #coef0 = [0,1];
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
        
        
        print "accuracy buck " + str(index) + " svm SVC POLYNOMIAN " +  str(accuracy) 
        
        clf = svm.SVC(kernel='sigmoid', coef0=4); #coef0 = [0,max(<x,y>)];
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
        
        print "accuracy buck " + str(index) + " svm SVC SIGMOID " +  str(accuracy) 
       
        
        clf = svm.SVC(kernel=ownKernelHamming);
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
              
        print "accuracy buck " + str(index) + " svm SVC OWNHAMMING " +  str(accuracy) 
        
        clf = svm.SVC(kernel=ownKernelEuclidean);
        clf.fit(ADA_Values_Train, ADA_Target_Train);
        
        accuracy = 0
        for i in range(len(ADA_Target_Data)):
            target_data = ADA_Values_Data[i];
            res = clf.predict(target_data);
            if (res == ADA_Target_Data[i]):
                accuracy += 1
        accuracy = float(accuracy)/float(len(ADA_Target_Data))
        
        print "accuracy buck " + str(index) + " svm SVC OWNEUCLIDEAN " +  str(accuracy)
        
    index+=buck;
    data_Bankruptcy_BayesTrain = []
    data_Bankruptcy_BayesData = []
    data_Bankruptcy_Total = []

print "========================================================================================"
    
print "representacion grafica de los kernels (se cogen los dos primeros elementos)"

ADA_Target_Train = []; #identifica las clases (B,NB)
ADA_Values_Train = []; #identifica los valores  (P,A,N)
        
#valores de testing
ADA_Target_Data = []; #identifica las clases (B,NB)
ADA_Values_Data = []; #identifica los valores (P,A,N)


for line in open( path_Qualitative_Bankruptcy,"r"):
    content = line.split(',');
    classe = content[6][:-1];
    data.append(classe);
    data.append(content[0]);
    data.append(content[1]);
    data.append(content[2]);
    data.append(content[3]);
    data.append(content[4]);
    data.append(content[5]);
    data_Bankruptcy_Total.append(data);
    data = [];

for i in range(len(data_Bankruptcy_Total)):
    data = data_Bankruptcy_Total[i]
    data_value = [];
    ADA_Target_Train.append(data_number_equivalency(data[0]));
    data_value.append(data_number_equivalency(data[1]));
    data_value.append(data_number_equivalency(data[2]));
    #data_value.append(data_number_equivalency(data[3]));
    #data_value.append(data_number_equivalency(data[4]));
    #data_value.append(data_number_equivalency(data[5]));
    #data_value.append(data_number_equivalency(data[6]));
    ADA_Values_Train.append(data_value);
    data_value = [];
    data = [];
    
    
    
'''
#representacion grafica
##########################################################################################
print "representacion Kernel Lineal"
clf = svm.SVC(kernel='linear');  
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);

#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('lineal KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################

#representacion grafica
##########################################################################################
print "representacion Kernel Polinomial"
clf = svm.SVC(kernel='poly', C=1e3, degree=4, coef0=1); 
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);

#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('Polynomian KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################

#representacion grafica
##########################################################################################
print "representacion Kernel Gaussian"
clf = svm.SVC(kernel='rbf', C=1e3, gamma=0.1); 
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);

#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('Gaussian KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################

#representacion grafica
##########################################################################################
print "representacion Kernel Sigmoid"
clf = svm.SVC(kernel='sigmoid', coef0=4);
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);

#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('Sigmoid KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################

#representacion grafica
##########################################################################################
print "representacion Kernel OwnHamming"
clf = svm.SVC(kernel=ownKernelHamming);  
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);

#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('OWNHAMMING KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################

#representacion grafica
##########################################################################################
print "representacion Kernel OwnEuclidean"
clf = svm.SVC(kernel=ownKernelEuclidean);
definition = clf.fit(ADA_Values_Train,ADA_Target_Train);
#Representacion grafica del SVM cogiendo dos valores de la representacion de 6 necesarios
h = .02  # step size in the mesh
x_min, x_max = min(ADA_Values_Train[:][0]) - 1, max(ADA_Values_Train[:][0]) + 1
y_min, y_max = min(ADA_Values_Train[:][1]) - 1, max(ADA_Values_Train[:][1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

my_cmap = pl.get_cmap('PiYG');
pl.pcolormesh(xx, yy, Z, cmap=my_cmap);
pl.scatter(ADA_Values_Train[:][0], ADA_Values_Train[:][1], c=ADA_Target_Train, cmap=my_cmap);
pl.title('OWNEUCLIDEAN KERNEL')
pl.axis('tight')
pl.show()
##########################################################################################
'''

#ADABOOST - BOOK
#######################################################################################
'''
T=50

def hti(atvl, val, igual):
    
    if ((igual & (val == atvl)) | (not igual & (val != atvl))):
        pred = +1
    else:
        pred = -1
    
    return pred


def error(Dt, cl, atvl, val, igual):
    s = 0
    for i in range(len(cl)):
        if (hti(atvl[i], val, igual) != int(cl[i])):
            s += Dt[i]
    
    return s


def WeakLearner(Dt, tr):
    ll = []
    for atr in range(len(tr)-1):
        for val in sorted(set(tr[atr+1])):
            et = error(Dt, tr[0], tr[atr+1], val, True)
            ll.append([et,atr+1,val,True])
            et = error(Dt,tr[0],tr[atr+1],val, False)
            ll.append([et,atr+1,val,False])
    return min(ll)

def textEx(ex, model):
    return sum([regla[0]*hti(ex[regla[2]],regla[3], regla[4]) for regla in model])


def adaBoostBookUOC(test, train):
     tt = list(zip(*train)) 
     m = len(train)        
     Dt = list(repeat((float(1)/float(m)),m))
     modelo = []
     for i in range(T):
         [et,atr,val,igual] = WeakLearner(Dt,tt)
         if et > 0.5:
             print "out of range!"
             break
         if et ==0:
             print "clasificado"
             break
         alfat = (float(1)/float(2))*math.log(float(1-et)/float(et))
         Dt2 = [Dt[j]*math.exp(-alfat * hti(tt[atr][j],val,igual)*float(train[j][0])) for j in range(m)]
         s = sum(Dt2)
         Dt = [(float(Dt2[j])/float(s)) for j in range(m)]
         modelo.append((alfat,et,atr,val,igual))
         
     prediccion = [textEx(ex,modelo) for ex in test]
     
     prec = (float(len(list(filter(lambda x: x[0] * float(x[1][0]) >0, zip(*[prediccion, test])))))/float(len(test)))
     print "prediccion " + str(prec) 
     
     return 0
'''
####################################################################################
