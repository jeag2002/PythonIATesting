'''
'PAC 4
'''

import numpy as np

import random, sys, math, operator

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


#VARIABLES GLOBALES
#############################################################################################
validacion_reg = [];
validacion_state = [];

entrenament_reg = [];
entrenament_state = [];

prova_reg = [];
prova_state = [];


inc = 0;    #reparte los datos entre validacion/entrenamiento y prova
flag = 0;   #cuenta los 200 elementos que representa un numero
level = 0;  #determina la clase.

C = 0;
gamma = 0;

path_mfeat_fac_path = ".././data/mfeat-fac.txt";
#############################################################################################

#ALGORITMO RECOCCION
#############################################################################################

def generateRandomCyGamma():
    C = math.pow(10,random.randrange(-10, 10));
    gamma = math.pow(10,random.randrange(-10, 10));
    return [C,gamma];


def tolerate(compare, accuracy, tolerancia, iteraciones):
    if (compare < accuracy):
        return True;
    else:
        print str(tolerancia) +  " " + str(iteraciones)
        factor = float(tolerancia)/float(iteraciones)
        valor = math.exp(float(compare-accuracy)/float(iteraciones*factor))
        return random.random() < valor
        
    


def evaluacion(clf, reg_prueba, state_prueba):
    
    accuracy = 0;
    
    for i in range(len(reg_prueba)):
        target_data = reg_prueba[i];
        res = clf.predict(target_data);
        
        if (res[0] == state_prueba[i]):
                accuracy += 1;
        
        
    accuracy = float(accuracy)/float(len(state_prueba));
        
    return accuracy;

def entreno(reg_entrenamiento, state_entrenamiento, inputC, inputGamma):
      
        clf = SVC(kernel='rbf', C=inputC, gamma=inputGamma);
        clf.fit(reg_entrenamiento, state_entrenamiento);
        return clf;

def recoccionSimuladaSVC(reg_entrenamiento, state_entrenamiento, reg_prueba, state_prueba, C, gamma, tolerancia, iteraciones):
    
    res = [];
    compare = 0;
    accuracy = 0;
    it = 0;
    
    Cprima = C;
    gammaprima = gamma;
    
    
    for i in range(iteraciones):
        
        clf = entreno(reg_entrenamiento, state_entrenamiento, Cprima, gammaprima);
        accuracy = evaluacion(clf, reg_prueba, state_prueba);
        if (compare <= accuracy):
            compare = accuracy
            C=Cprima;
            gamma=gammaprima;
            it = i;
            
        else:
            Cprima=C
            gammaprima=gamma
        
        [Cprima,gammaprima] = generateRandomCyGamma();
            
    return [C,gamma,it,compare];

#############################################################################################



def geneticSVC(reg_entrenamiento, state_entrenamiento, reg_prueba, state_prueba, CMin, CMax, gammaMin, gammaMax, iteraciones):
    
    C = 0
    gamma = 0
    
    Cprima = CMin
    gammaPrima = gammaMin
    
    compare = 0
    accuracy = 0
    it = 0
    
    for i in range(iteraciones):
        clf = entreno(reg_entrenamiento, state_entrenamiento, Cprima, gammaPrima);
        accuracy = evaluacion(clf, reg_prueba, state_prueba);
        
        if (compare < accuracy):
            C = Cprima
            gamma = gammaPrima
            compare = accuracy
            it = i
        
        if gammaPrima < gammaMax:
            gammaPrima = gammaPrima*10; #Evolucion
        else:   
            gammaPrima = gammaMin;  #Mutacion
            if Cprima < CMax:
                Cprima = Cprima*10; #Evolucion
            else:
                Cprima = CMin; #Mutacion
                
    return [C, gamma, it, compare]


#ALGORITMO GENETICO
#############################################################################################

#############################################################################################


print "EJERCICIO-1: preprocesamiento"
print "======================================================================================================"

for line in open( path_mfeat_fac_path,"r"):
    
    data = []
    content = line.split();
    
    for x in range(len(content)):
        data.append(float(content[x]))
    
    if (flag >= 200):
        flag = 0;
        level += 1;
    else:
        flag += 1;
    
    if (inc == 0):
       validacion_reg.append(data);
       validacion_state.append(level);
       inc += 1;
       
    elif (inc == 1):
       entrenament_reg.append(data);
       entrenament_state.append(level);
       inc +=1;
        
    else:
        prova_reg.append(data);
        prova_state.append(level);
        inc = 0;
print "======================================================================================================"

print "EJERCICIO-2:"
print "======================================================================================================"

'''
d = np.array(entrenament_reg).reshape(len(entrenament_reg),216);
X_entrenamiento_array = d-d.mean(0)
#X_entrenamiento_array = np.dot(X_entrenamiento_array.transpose(),X_entrenamiento_array)
X_entrenamiento = X_entrenamiento_array.tolist();


d2 = np.array(prova_reg).reshape(len(prova_reg),216);
X_prova_array = d2-d2.mean(0)
#X_validacion_array = np.dot(X_validacion_array.transpose(),X_validacion_array)
X_prova = X_prova_array.tolist();
'''

scaler = StandardScaler();
X_entrenamiento = scaler.fit_transform(entrenament_reg)
X_prova = scaler.fit_transform(prova_reg)
X_validacion = scaler.fit_transform( validacion_reg)


C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=entrenament_state, n_folds=3)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_entrenamiento, entrenament_state)

print "The best classifier is: " + str(grid.best_estimator_)


C_2d_range = [1, 1e2, 1e4]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_prova, prova_state)
        classifiers.append((C, gamma, clf))

print "Others Classifiers " +str(classifiers)

        
print "======================================================================================================"

print "EJERCICIO-3: recoccion simulada"
print "======================================================================================================"

[COut,gammaOut,it,accuracyOut] = recoccionSimuladaSVC(X_entrenamiento,  entrenament_state, X_prova, prova_state, 10.0, 0.001, 0.1, 50);
print "PROVA ::= C " + str(COut) + " gamma " + str(gammaOut) + " it " + str(it) + " accuracy " + str(accuracyOut)

[COut,gammaOut,it,accuracyOut] = recoccionSimuladaSVC(X_entrenamiento,  entrenament_state, X_validacion, validacion_state, 10.0, 0.001, 0.1, 50);
print "VALIDATE ::= C " + str(COut) + " gamma " + str(gammaOut) + " it " + str(it) + " accuracy " + str(accuracyOut)


print "======================================================================================================"



print "EJERCICIO-4: algoritmo genetico"
print "======================================================================================================"

[COut,gammaOut,it,accuracyOut] = geneticSVC(X_entrenamiento,  entrenament_state, X_prova, prova_state, 1, 1000, 0.00001, 1, 50);
print "PROVA ::= C " + str(COut) + " gamma " + str(gammaOut) + " it " + str(it) + " accuracy " + str(accuracyOut)

[COut,gammaOut,it,accuracyOut] = geneticSVC(X_entrenamiento,  entrenament_state, X_validacion, validacion_state, 1, 1000, 0.00001, 1, 50);
print "VALIDATE ::= C " + str(COut) + " gamma " + str(gammaOut) + " it " + str(it) + " accuracy " + str(accuracyOut)

print "======================================================================================================"


