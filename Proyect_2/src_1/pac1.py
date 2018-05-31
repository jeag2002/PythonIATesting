'''
'PAC-1
'''


import random;
import math;

from sklearn import metrics;
from random import sample
from itertools import repeat

KMEAN=4; 
BIG_NUMBER = math.pow(10, 10)
MAX_ITER=10000;


'''
' VARIABLES GLOBALES
'''
data_vector = []; #vector
hotel_stats = []; #data
labels_true = []; #clusteres de datos
labels_true_UOC = []; #clusteres de datos UOC

dic = {};		  #diccionario de entrada para funcion KMEANS de la UOC

kmeans_output_data = (); #secuencia de salida para la funcion KMEANS de la UOC
centroid_hotel_stats = []; #centroides iniciales
data_medias_ponderadas = []; #medias ponderadas, ejercicio 4

path_read = ".././data_1/hotels.data";
path_write = ".././data_1/favorits_pr.data";

'''
' FUNCIONES AUXILIARES: DISTANCIAS
'''
#########################################################################################################
#########################################################################################################

#calculo de la media de una linea
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)
#end def

#calculo del coeficiente de PEARSON a partir de dos listas.

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    
    if ((xdiff2==0) | (ydiff2==0)):
        return 0
    else:
        return diffprod / math.sqrt(xdiff2 * ydiff2)
    #end if

'''
'Calculo de coeficiente de correlacion de PEARSON definido en wikipedia
'http://es.wikipedia.org/wiki/Coeficiente_de_correlaci%C3%B3n_de_Pearson
'''

def pearson_def_wiki(x,y):
	
    if (len(x) != len(y)):
        return 0
    n = len(x);
    sum_x = sum(x);
    sum_y = sum(y);
    sum_xy = 0;
    sum_x_square = 0;
    sum_y_square = 0;
    num = 0;
    den = 0;
    for idx in range(n):
		sum_xy += x[idx]*y[idx];
		sum_x_square += math.pow(x[idx],2);
		sum_y_square += math.pow(y[idx],2);
	#end for
    den = math.sqrt(n*sum_x_square - math.pow(sum_x,2))*math.sqrt(n*sum_y_square - math.pow(sum_y,2));
    num = n*sum_xy - sum_x*sum_y;
    
    if (den == 0):
		return 0;
    else:    
        return num/den;
	#end if
#end def

#definicion de la distancia EUCLIDEA entre el centroide - punto (con listas)
def euclidean_distance(centroid, dataPoint):
    
    if (len(centroid) != len(dataPoint)):
        return 0;
    else:
        
        data = 0;
        
        for i in range(len(centroid)):
            data += math.pow((float(centroid[i]) - float(dataPoint[i])), 2);   
        #end for
        
        return  int(math.sqrt(data));
	#end if
#end def 

#definicion de la distancia EUCLIDEA entre el centroide - punto (con diccionarios)
def euclideanDist(dic1, dic2):
    
    sum2 = sum([pow(dic1[elem]-dic2[elem], 2) for elem in dic1 if elem in dic2])
    
    return math.sqrt(sum2);
#end def


#definicion de la distancia dic
def euclideanDistDic(centroidDic, dataPointDic):
    if (len(centroidDic.values()) != len(dataPointDic.values())):
        return 0;
    else:
        data = 0;
        
        for i in range(len(centroidDic.values())):
            data += math.pow((float(centroidDic.values()[i]) - float(dataPointDic.values()[i])), 2);   
        #end for
        return  math.sqrt(data);
    #end if  
#end def

def pearsonDistDic(centroidDic, dataPointDic):
    return pearson_def_wiki(centroidDic.values(),dataPointDic.values());


#definicion de la distancia EUCLIDEA entre el centroide - punto (con diccionarios)
def euclideanSimilarity(dic1, dic2):
    return 1/(1+euclideanDistDic(dic1, dic2))
#end def


def pearsonSimilarity(dic1, dic2):
    return 1/(1+pearsonDistDic(dic1, dic2))


#########################################################################################################
#########################################################################################################

'''
' FUNCIONES AUXILIARES K-MEANS
'''
#########################################################################################################
#########################################################################################################

'''
' FUNCION AUXILIAR K-MEANS (Implementada por mi)
' k-means http://mnemstudio.org/ai/cluster/k_means_python_ex1.txt
'''

#recalculo de los centroides 
def recalculate_centroids():
    totalA = 0;
    totalB = 0;
    totalC = 0;
    totalD = 0;
    totalE = 0;
    totalF = 0;
    totalG = 0;
    totalH = 0;

    totalInCluster = 0;
    vectorData = [];
    
    for j in range(KMEAN):
        for k in range(len(labels_true)):
            if (labels_true[k] == j):    
                vectorData = hotel_stats[k];
                totalA += float(vectorData[0]);
                totalB += float(vectorData[1]);
                totalC += float(vectorData[2]);
                totalD += float(vectorData[3]);
                totalE += float(vectorData[4]);
                totalF += float(vectorData[5]);
                totalG += float(vectorData[6]);
                totalH += float(vectorData[7]);
                totalInCluster += 1;
                vectorData = [];
            
        if (totalInCluster > 0):
            
            vectorCentroid = [];
            vectorCentroid.append((float(float(totalA)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalB)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalC)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalD)/float(totalInCluster))));
            
            vectorCentroid.append((float(float(totalE)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalF)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalG)/float(totalInCluster))));
            vectorCentroid.append((float(float(totalH)/float(totalInCluster))));
            
            centroid_hotel_stats[j] = vectorCentroid;
            
    return

#reorganizacion de los clusteres
def update_clusters():
    
    isStillMoving = 0
    
    for i in range(len(hotel_stats)):
        
        bestMinimum = BIG_NUMBER
        currentCluster = 0
        
        for j in range(KMEAN):
            distance = euclidean_distance(centroid_hotel_stats[j], hotel_stats[i])
            if(distance < bestMinimum):
                bestMinimum = distance
                currentCluster = j
        
        labels_true[i] = currentCluster;
        
        if(labels_true[i] == -1 or labels_true[i] != currentCluster):
            isStillMoving = 1
            
    return isStillMoving
#end def                     
                                                      
#calculo kmeans 
def perform_kmeans():
    isStillMoving = 1
    
    while(isStillMoving):
        recalculate_centroids();
        isStillMoving = update_clusters(); 
        
    return
#end def


'''
' FUNCION AUXILIAR K-MEANS (entregado por la UOC)
'''
#diccionario => datos de entrada
#k => numero de centroides (KMEANS)
#maxit => numero maximo de iteraciones
def kmeans_dictio(dictionary, k, maxit, similarity = euclideanSimilarity):
    
   centroids = [dictionary[x] for x in sample(dictionary.keys(), k)]
      
   previous   = {}
   assignment = {}
   

   for it in range(maxit):
       
       #key1 ;= identificador de hoteles
       
       for key1 in dictionary:
           simils = map(similarity,repeat(dictionary[key1],k), centroids) 
           #repeat := repite el elemento dictionary[key], KMEANS = 4 veces
		   #map := por cada elemento de repeat, se le aplica similarity con
		   #centroids de parametro de entrada extra y lo vuelca en simils
           assignment[key1] = simils.index(max(simils))           

       if previous == assignment:
           break
       
       previous.update(assignment)
       
       values   = {x : {} for x in range(k)}
       counters = {x : {} for x in range(k)}
       
       for key1 in dictionary:
           group = assignment[key1]
           
           for key2 in dictionary[key1]:
               
               if not values[group].has_key(key2):
                   values   [group][key2] = 0
                   counters [group][key2] = 0
                   
               values  [group][key2] += dictionary[key1][key2]
               counters[group][key2] += 1
        
       centroids = []
       for group in values:
           centr = {}
           for key2 in values[group]:
               centr[key2] = values[group][key2] / counters[group][key2]
           centroids.append(centr)
       
       if None in centroids: break

   print "iteraciones " + str(it)
   
   return (assignment, centroids)
#end def

#########################################################################################################
#########################################################################################################

print "EJERCICIO 1 - preprocesado: "
print "****************************************************************************"

content = [];
vector = [];
dic_data = {};

for line in open( path_read,"r"):
    
    
    content = line.split();
    hotel = content[0];
    
    if hotel not in data_vector:
        #listas
        vector.append(int(content[2]));
        vector.append(int(content[3]));
        vector.append(int(content[4]));
        vector.append(int(content[5]));
        vector.append(int(content[6]));
        vector.append(int(content[7]));
        vector.append(int(content[8]));
        vector.append(int(content[9]));
        
        data_vector.append(hotel);
        hotel_stats.append(vector);
        
		#diccionario (set solo valoraciones generales)
        dic_data = {1:content[2],2:content[3],3:content[4],4:content[5],5:content[6],6:content[7],7:content[8],8:content[9]};
        dic.update({int(hotel):dic_data});
		
		#inicializacion del vector de clusteres.
        labels_true.append(-1);
        vector = [];
        dic_data = {};
	
    else:
        index = data_vector.index(hotel);
        vector = hotel_stats[int(index)];
        vector[0] = int((float(vector[0])+float(content[2]))/float(2));
        vector[1] = int((float(vector[1])+float(content[3]))/float(2));
        vector[2] = int((float(vector[2])+float(content[4]))/float(2));
        vector[3] = int((float(vector[3])+float(content[5]))/float(2));
        vector[4] = int((float(vector[4])+float(content[6]))/float(2));
        vector[5] = int((float(vector[5])+float(content[7]))/float(2));
        vector[6] = int((float(vector[6])+float(content[8]))/float(2));
        vector[7] = int((float(vector[7])+float(content[9]))/float(2));
        hotel_stats[index] = vector;
        vector = [];
        
        #diccionario (set solo valoraciones generales)
        dic_data = dic[int(hotel)];
        dic_data[1] = int((float(dic_data[1])+float(content[2]))/float(2));
        dic_data[2] = int((float(dic_data[2])+float(content[3]))/float(2));    
        dic_data[3] = int((float(dic_data[3])+float(content[4]))/float(2));    
        dic_data[4] = int((float(dic_data[4])+float(content[5]))/float(2));    
        dic_data[5] = int((float(dic_data[5])+float(content[6]))/float(2));    
        dic_data[6] = int((float(dic_data[6])+float(content[7]))/float(2));    
        dic_data[7] = int((float(dic_data[7])+float(content[8]))/float(2));    
        dic_data[8] = int((float(dic_data[8])+float(content[9]))/float(2));    
        	
        dic.update({int(hotel):dic_data});
        dic_data = {};
		
    #end if

print "****************************************************************************"

print "EJERCICIO 1 - RESULTADOS: "
print "****************************************************************************"

print "preprocesado MIO:"
print "hoteles             MIO ::= " + str(data_vector[ : ])
print "valoraciones medias MIO ::= " + str(hotel_stats[ : ]);

print "preprocesado UOC:"
print "hoteles             UOC ::= " + str(dic.keys());
print "valoraciones medias UOC ::= " + str([dic[x].values() for x in dic.keys()]);

print "****************************************************************************"


print "EJERCICIO 2 - KMEANS-4: "
print "****************************************************************************"

'''
determina los centroides
'''
     
lim_00 = [0,0,0,0,0,0,0,0];
lim_01 = [0,0,0,0,10,10,10,10];
lim_10 = [10,10,10,10,0,0,0,0];
lim_11 = [10,10,10,10,10,10,10,10];

bestMinimum_00 = BIG_NUMBER;
bestMinimum_01 = BIG_NUMBER;
bestMinimum_10 = BIG_NUMBER;
bestMinimum_11 = BIG_NUMBER;

index_00 = 0;
index_01 = 0;
index_10 = 0;
index_11 = 0;


#calculo de los centroides (4 centroides) ==> puede ser cualquier valor, establecemos los valores mas cercanos a las esquinas lim_00, lim_01, lim_10, lim_11
for j in range(len(hotel_stats)):
    
    vectorCentroide = hotel_stats[j];
    distancia = pearson_def(lim_00, vectorCentroide);
    
    if (distancia < bestMinimum_00):
        index_00 = j;
        bestMinimum_00 = distancia;
    
    distancia = pearson_def(lim_01, vectorCentroide);
    
    if (distancia < bestMinimum_01):
        index_01 = j;
        bestMinimum_01 = distancia;
        
        
    distancia = pearson_def(lim_10, vectorCentroide);
    
    if (distancia < bestMinimum_10):
        index_10 = j;
        bestMinimum_10 = distancia;

    distancia = pearson_def(lim_11, vectorCentroide);
    
    if (distancia < bestMinimum_11):
        index_11 = j;
        bestMinimum_11 = distancia;
#end for

#asignamos a 4 hoteles los cuatro centroides.
labels_true[index_00]=0;
labels_true[index_01]=1; 
labels_true[index_10]=2; 
labels_true[index_11]=3; 

centroid_hotel_stats.append(hotel_stats[index_00]);
centroid_hotel_stats.append(hotel_stats[index_01]);
centroid_hotel_stats.append(hotel_stats[index_10]);
centroid_hotel_stats.append(hotel_stats[index_11]);

'''
'efectua el calculo de kmeans propuesto por mi
'''
perform_kmeans();

'''
'efectua el calculo de kmeans propuesto por la UOC
'''
kmeans_output_data = kmeans_dictio(dic, KMEAN, MAX_ITER);

print "****************************************************************************"

print "EJERCICIO 2 - RESULTADOS: "
print "****************************************************************************"
print "proceso KMEANS MIO: " + str(labels_true[ : ]);
print "proceso KMEANS UOC: " + str(kmeans_output_data[0].values());
print "****************************************************************************"

print "EJERCICIO 3 - DISTANCIAS: "
print "****************************************************************************"

for i in range(len(labels_true)):
    labels_true[i] = int(labels_true[i]);
    
labels_pred = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]

data = metrics.adjusted_rand_score(labels_true,labels_pred)

labels_true_uoc = kmeans_output_data[0].values();

data_1 = metrics.adjusted_rand_score(labels_true_uoc,labels_pred);


print "****************************************************************************"

print "EJERCICIO 3 - RESULTADOS: "
print "****************************************************************************"

print "adjusted rand scored (metrics) MIO " + str(data)
print "adjusted rand scored (metrics) UOC " + str(data_1)

print "****************************************************************************"



print "EJERCICIO 4 - MEDIAS PONDERADAS: "
print "****************************************************************************"


content_media = [];
content_media_hotel = [];

vector_media = [];
index_data = 0;                   #indice dentro del vector, volcado del fichero
index_data_hotel = 0              #indice dentro del vector, volcado del fichero, busqueda de hoteles

index_user = 0;                   #indice del usuario a tratar.
index_hotel = 0;                  #indice del hotel a tratar

vector_hotel_x_usuario = [];      #hoteles valorados por un usuario
vector_hotel_x_usuario_pond = []; #valoraciones ponderadas de un usuario

num = 0;
den = 0;

valorHotelUsuario = 0;

#inicializamos rangos 
for userId in range(100):
    data_medias_ponderadas.append(0);

#volcamos toda la informacion del sistema en un vector.
for line_media in open( path_read,"r"):
    content_media = line_media.split();
    content_media[0] = int(content_media[0]);
    content_media[1] = int(content_media[1]);
    content_media[2] = float(content_media[2]);
    content_media[3] = float(content_media[3]);
    content_media[4] = float(content_media[4]);
    content_media[5] = float(content_media[5]);
    content_media[6] = float(content_media[6]);
    content_media[7] = float(content_media[7]);
    content_media[8] = float(content_media[8]);
    content_media[9] = float(content_media[9]);
    vector_media.append(content_media);
#end if

for index_data in range(len(vector_media)):
    
    content_media = vector_media[index_data];
    index_user = content_media[1]; #obtenemos el usuario
    index_hotel = content_media[0]; #obtenemos el indice de hoteles
    
    int_index_user = int(int(index_user)-1) #obtenemos el id del usuario
    
    if (data_medias_ponderadas[int_index_user]==0): #usuario ya tratado?
        '''
        'tratar hoteles del usuario
        '''
        if (index_hotel not in vector_hotel_x_usuario):
            
            vector_hotel_x_usuario.append(index_hotel); #hotel
            
			#valoracion general hotel index_hotel para el usuario del index_user

            valorHotelUsuario = content_media[2];  
                
            #valorHotelUsuario = average(content_media[2:len(content_media)]);
			#valorHotelUsuario = pearson_def(content_media[2:len(content_media)],content_media[2:len(content_media)]);
			#valorHotelUsuario = pearson_def_wiki(content_media[2:len(content_media)],content_media[2:len(content_media)]);
            #valorHotelUsuario = euclidean_distance(content_media[2:len(content_media)],content_media[2:len(content_media)]);
            '''
            'tratar usuarios que hayan valorado este mismo hotel
            '''
            
            for index_data_hotel in range(len(vector_media)):
                
                 content_media_hotel = vector_media[index_data_hotel];
                 
                 if ((content_media_hotel[1]!=index_user) & (content_media_hotel[0]==index_hotel)):
                     
                     #pearson_def (valoracion de un mismo hotel con respecto a otro usuario que ha hecho esta valoracion)
                     
                     #numerador
                     num += pearson_def_wiki(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)])* valorHotelUsuario;
                     #num += pearson_def_wiki(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)])* valorHotelUsuario;
					 #num += euclidean_distance(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)])* valorHotelUsuario;
                     #denominador
                     den += pearson_def_wiki(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)]);
                     #den += pearson_def_wiki(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)]);
					 #den += euclidean_distance(content_media[2:len(content_media)],content_media_hotel[2:len(content_media_hotel)]);
                 #end if    
            #end for 
            
            '''
            'calcular ponderacion
            '''
                     
            if (den == 0):
                num = 0
            else:
                num = float(float(num)/float(den));
            #end if
            
            
            #valoracion ponderada del hotel hecho por el usuario
            vector_hotel_x_usuario_pond.append(sum);
            num = 0; 
            den = 0;
        
        #end if
        
        '''
        'obtener hotel mejor valorado
        '''
        ponderacion = vector_hotel_x_usuario_pond[0];
        index_ponderacion = 0;
        
        for index_data_hotel in range(len(vector_hotel_x_usuario_pond)):
            
            if(ponderacion < vector_hotel_x_usuario_pond[index_data_hotel]):
                index_ponderacion = index_data_hotel;
            #end if
        #end if
        
        if (index_ponderacion > 0):
            data_medias_ponderadas[int_index_user] = vector_hotel_x_usuario[index_ponderacion];
        else:
            data_medias_ponderadas[int_index_user] = index_hotel
        
        
    #end if
#end for    

print "****************************************************************************"

print "EJERCICIO 4 - RESULTADOS: "
print "****************************************************************************"

print "medias ponderadas por usuario"
print data_medias_ponderadas[ : ];

fwrite = open(path_write,'w')

userId = 0
for userId in range(100):
    dataInt = userId+1
    fwrite.write("%d\t%d\n" % (dataInt, data_medias_ponderadas[userId]));
fwrite.close()

print "****************************************************************************"