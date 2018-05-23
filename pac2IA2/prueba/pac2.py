'''
'PAC-2
'''

'''
Problemas: (Python v2.7 w32; SO Windows 7 64 bits ultimate)

reinstalar numpy-1.8.1.py win 32 
intalar matplotlib 1.3 win 32

poner en ...C:\Python27\Lib\site-packages\mpl_toolkits archivo __init__.py vacio
para que reconozca las librerias mpl_toolkits.mplot3d 
(http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe)

instalar six-1.6.1 python 2.7 win 32
instalar pyparsing-1.5.7 python 2.7 win 32
instalar python-dateutil-2.2 python 2.7 win 32
instalar scikit-learn-0.14.1 python 2.7 win 32

http://books.google.es/books?id=nwZ6AgAAQBAJ&pg=PA334&lpg=PA334&dq=2.+mfeat-fac:+216+profile+correlations;&source=bl&ots=bWl0FLGWRC&sig=wAQXGTWPVdqzU72-U_ICHJpsk1g&hl=en&sa=X&ei=wEVMU_bGIqeY1AWz6YCQBA&ved=0CFAQ6AEwBQ#v=onepage&q=2.%20mfeat-fac%3A%20216%20profile%20correlations%3B&f=false
http://stackoverflow.com/questions/13224362/pca-analysis-with-python
http://www.lfd.uci.edu/~gohlke/pythonlibs/
https://pypi.python.org/pypi/scikit-learn/0.14.1#downloads
'''
import os
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import math;
# from scikits.learn.lda import LDA no existe en la version 2.7
from sklearn.lda import LDA
import mdp

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

np.set_printoptions(precision = 3)

#Variables globales
###################################################################################
path_fac = ".././data/mfeat-fac.txt";
path_pix = ".././data/mfeat-pix.txt";

MAX_IMG = 5;
MAX_PIX = 240;
MAX_FAC = 216;

NPIXFLAT = [];
NPIX = []; 
NCORRFLAT = [];
NCORR = [];

content = [];
linea = 0;
###################################################################################

print "EJERCICIO 0 - procesado de los ficheros de entrada"
#http://pendientedemigracion.ucm.es/info/Astrof/POPIA/asignaturas/ana_dat_est/tema09_x2.pdf
print "****************************************************************************"

#fichero de pixeles
for line in open( path_pix,"r"):
    content = line.split();
    for x in range(len(content)):
        #NPIXFLAT.append(float(pow(2,int(content[x]))-1))
        NPIXFLAT.append(float(content[x]))

content = [];
linea = 0;

#fichero de correlaciones
for line in open( path_fac,"r"):
    content = line.split();
    for x in range(len(content)):
        NCORRFLAT.append(int(content[x]))

print "****************************************************************************" 


print "EJERCICIO 2 - PCA: (ejercicio de clasificacion)"
print "****************************************************************************"
#http://stackoverflow.com/questions/4801259/whats-wrong-with-my-pca
#http://stackoverflow.com/questions/4823223/numpy-eig-and-the-percentage-of-variance-in-pca



dx = np.array(NPIXFLAT).reshape(2000,240);
d = dx;

fig1 = plot.figure()
sp = fig1.gca()
sp.scatter(d[:,2],d[:,3])
plot.xlim([-7,7])
plot.ylim([-7,7])


#sp = fig1.gca(projection = '3d')
#sp.scatter(d[:,0],d[:,1],d[:,2])
plot.show()

#manera de calcular valores propios, vectores propios de la matriz de covarianza, utilizando eig (buscado en internet)
#data = (d - d.mean(axis=0)) / d.std(axis=0)
#C = np.corrcoef(d, rowvar=0)
#valp1,vecp1 = np.linalg.eig(C)

#manera de calcular valores propios, vectores propios de la matriz de covarianza, utilizando svd (buscado en internet)
#d = (d - d.mean(axis=0)) / d.std(axis=0)
#C = np.corrcoef(d, rowvar=0)
#u, valp1,vecp1 = np.linalg.svd(C)

#manera de calcular valores propios, vectores propios de la matriz de covarianza. Propuesto por la UOC
d1 = d-d.mean(0)
matcov = np.dot(d1.transpose(),d1)
valp1,vecp1 = np.linalg.eig(matcov)

#manera de calcular valores propios, vectores propios de la matriz de covarianza, utilizando eig (buscado en internet)
#d -= np.mean(d, axis=0)
#d = np.corrcoef(d, rowvar=0)
#valp1,vecp1 = np.linalg.eig(d)

#ejercicio 2

print "valores propios : " + str(valp1.tolist());
#print "vectores propios : " + str(vecp1.tolist());

#para  NPIXFLAT numero de elementos minimo para conseguir una varianza del 95% son 88 elementos.
num = sum([valp1[elem] for elem in range(88)]);
value =  sum([valp1[elem] for elem in range(len(valp1))]);  
num = num/value * 100;
print "proporcionalidad de la varianza: " +  str(num) + " % "

ind_creixent = np.argsort(valp1)
ind_decre = ind_creixent[::-1]
val_decre = valp1[ind_decre]
vec_decre = vecp1[:,ind_decre]
pylab.plot(val_decre,'o-')

d_PCA = np.zeros((d.shape[0], d.shape[1]))

for i in range (d.shape[0]):
    for j in range (d.shape[1]):
        d_PCA[i,j] = np.dot(d[i,:], vecp1[:,j])

d_recon = np.zeros((d.shape[0], d.shape[1]))
for i in range (d.shape[0]):
    for j in range (d.shape[1]): 
        d_recon[i]+= d_PCA[i,j]*vecp1[:,j]

np.allclose(d,d_recon)

d_PCA2 = np.zeros((d.shape[0],2))
for i in range(d.shape[0]):
    for j in range(2):
          d_PCA2[i,j]  = np.dot(d[i,:], vec_decre[:,j]) 

d_recon2 = np.zeros((d.shape[0],d.shape[1]))
for i in range(d.shape[0]):
    for j in range(2):
          d_recon2[i] += d_PCA2[i,j] * vec_decre[:,j]
          
fig2 = plot.figure()

#sp2 = fig2.gca(projection = '3d')
#sp2.scatter(d_recon2[:,0],d_recon2[:,1],d_recon2[:,2],c='r',marker='x')
#sp2.scatter(d_recon2[:,3],d_recon2[:,4],d_recon2[:,5],c='b',marker='o')

#ejercicio 3
sp2 = fig2.gca()
sp2.scatter(d_recon2[:,2],d_recon2[:,3],c='b',marker='o')
sp2.scatter(d_recon2[:,0],d_recon2[:,1],c='r',marker='x')
plot.show()

print "****************************************************************************"


'''
print "EJERCICIO 2.1 - PCA: (ejercicio de clasificacion)"
print "****************************************************************************"
#http://mdp-toolkit.sourceforge.net/

dx = np.array(NPIXFLAT).reshape(2000,240);
d = dx[:,0:3];
y = mdp.pca(d);
print y 

print "****************************************************************************"
'''

    
'''     
#ejercicio 3.6     
#http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html

A = np.array(NPIXFLAT).reshape(5,240)
numdata,dim = A.shape
in_media = A.mean(axis=0);
for i in range(numdata):
    A[i] -= in_media;

M = np.dot(A,A.T);
lam,vec = np.linalg.eigh(M)
aux = np.dot(A.T,vec).T

V = aux[::-1]
S = np.sqrt(lam)[::-1]

in_media = in_media.reshape(15,16)
pylab.plot(S[0:10],'o-');

mode1 = V[0].reshape(15,16)
mode2 = V[1].reshape(15,16)

fig1 = pylab.figure()
fig1.suptitle('Imagen media')
pylab.gray()
pylab.imshow(in_media)

fig2 = pylab.figure()
fig2.suptitle('Primer modo PCA')
pylab.gray()
pylab.imshow(mode1)

fig3 = pylab.figure()
fig3.suptitle('Segundo modo PCA')
pylab.gray()
pylab.imshow(mode2)

pylab.show()
'''

print "EJERCICIO 4 - LDA: (ejercicio de discriminacion)"
print "****************************************************************************"

#urls:
#http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html
#http://research.cs.tamu.edu/prism/lectures/pr/pr_l10.pdf

X = np.array(NPIXFLAT).reshape(2000,240)
VectArray = np.array_split(X,2);

#escojo solo dos variables. hacer un bucle con 216 elementos puede ser impracticable.
X1 = VectArray[0][:,0:2]
X2 = VectArray[1][:,0:2]


fig1 = pylab.figure()
pylab.scatter(X1[:,0],X1[:,1],marker='^',c='r')
pylab.scatter(X2[:,0],X2[:,1],marker='o',c='b',hold='on')
pylab.legend(('Grupo 1','Grupo 2'))

XT=np.concatenate((X1,X2))
label1 = np.ones(X1.shape[0]) 
label2 = 2*np.ones(X2.shape[0])
labelT = np.concatenate((label1,label2))

clf = LDA()
clf.fit(XT,labelT)
LDA(priors=None)

print "definicion linea divisoria: " 
print str(clf.means_)

punto_1_x = clf.means_[0][0]
punto_1_y = clf.means_[0][1]

punto_2_x = clf.means_[1][0]
punto_2_y = clf.means_[1][1]


max_data = max(XT[:,0:1])
min_data = min(XT[:,0:1])

x = np.linspace(min_data,max_data,10);

pendiente = float(float(punto_2_y -punto_1_y)/float(punto_2_x-punto_1_x))
y = float(punto_1_y) +  pendiente * x - pendiente *punto_1_x 
pylab.plot(x,y, '-', color='black', markersize=1)



group_1 = 0
group_2 = 0

fig2 = pylab.figure()
for i in range(-1000,1000,50):
    for k in range(-1000,1000,50):
        p=clf.predict([[i,k]])
        #print i,k,p
        if p == 1:
            group_1 +=1
            pylab.scatter(i,k,s=20,marker='o',c='r',hold='on')
        else:
            group_2 +=1
            pylab.scatter(i,k,s=20,marker='o',c='b',hold='on')

pylab.axis([-1000,1000,-1000,1000])

group_1 = float(float(group_1)/float(1600))*100.0
group_2 = float(float(group_2)/float(1600))*100.0

print "estadistica clasificacion 1600 numeros"
print "grupo 1: " + str(group_1) + "%"
print "grupo 2: " + str(group_2) + "%"

pylab.show()    

print "****************************************************************************"