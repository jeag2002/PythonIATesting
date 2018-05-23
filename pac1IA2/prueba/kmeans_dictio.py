#! /usr/bin/env python
# -*- coding: utf-8 -*-


from random import sample
from itertools import repeat
from math import sqrt


def euclideanDist(dic1, dic2):
    # Compute the sum of squares of the elements common
    # to both dictionaries
    sum2 = sum([pow(dic1[elem]-dic2[elem], 2)
                for elem in dic1 if elem in dic2])
    return sqrt(sum2)

def euclideanSimilarity(dic1, dic2):
    return 1/(1+euclideanDist(dic1, dic2))
    
def pearsonCoeff(dic1, dic2):
    # Retrieve the elements common to both dictionaries
    commons  = [x for x in dic1 if x in dic2]
    nCommons = float(len(commons))

    # If there are no common elements, return zero; otherwise
    # compute the coefficient
    if nCommons==0:
        return 0

    # Compute the means of each dictionary
    mean1 = sum([dic1[x] for x in commons])/nCommons
    mean2 = sum([dic2[x] for x in commons])/nCommons

    # Compute numerator and denominator
    num  = sum([(dic1[x]-mean1)*(dic2[x]-mean2) for x in commons])
    den1 = sqrt(sum([pow(dic1[x]-mean1, 2) for x in commons]))
    den2 = sqrt(sum([pow(dic2[x]-mean2, 2) for x in commons]))
    den  = den1*den2

    # Compute the coefficient if possible or return zero
    if den==0:
        return 0

    return num/den
    
# Given a dictionary like {key1 : {key2 : value}} it computes k-means
# clustering, with k groups, executing maxit iterations at most, using
# the specified similarity function.
# It returns two things (as a tuple):
# -{key1:cluster number} with the cluster assignemnts (which cluster
#  does each element belong to
# -[{key2:values}] a list with the k centroids (means of the values
#  for each cluster.
# Recall that input dictionary can be sparse, and that will be reflected
# on the centroids list.
def kmeans_dictio(dictionary, k, maxit, similarity = euclideanSimilarity):
    
   # First k random points are taken as initial centroids.
   # Each centroid is {key2 : value}
   centroids = [dictionary[x] for x in sample(dictionary.keys(), k)]
   
   # Assign each key1 to a cluster number 
   previous   = {}
   assignment = {}
   
   # On each iteration it assigns points to the centroids and computes
   # new centroids
   for it in range(maxit):

       # Assign points to the closest centroids
       for key1 in dictionary:
           simils = map(similarity,repeat(dictionary[key1],k), centroids)
           assignment[key1] = simils.index(max(simils))           

       # If there are no changes in the assignment then finish
       if previous == assignment:
           break
       previous.update(assignment)
        
       # Recompute centroids: annotate each key values at each centroid
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
        
       # Compute means (new centroids)
       centroids = []
       for group in values:
           centr = {}
           for key2 in values[group]:
               centr[key2] = values[group][key2] / counters[group][key2]
           centroids.append(centr)
       
       if None in centroids: break

    
       
   return (assignment, centroids)


