'''
Applies Naive Bayes algorithm to train and test parameters and
return the accuracy. Classes have to be in the first column. All
the features have to be nominals.
'''

from functools import reduce
from collections import Counter
from copy import deepcopy
from sys import exit

# function declarations

def NxikCounter(v, k, cs):
   return {c: dict(Counter(
      map(lambda x: v[x[0]],
          list(filter(lambda e: e[1] == c,
                      enumerate(list(k))))))) for c in cs}

def Pxik(Nk, ntr, atr, N, k):
   if atr in N[k]:
      return N[k][atr] / Nk[k]
   else:
      return Nk[k] / ntr ** 2

def classify(t, Nk, Nxik, ntr):
   nts = len(t)
   l = [(k , Nk[k] / ntr *
        reduce(lambda x, y: x * y,
               map(Pxik, [Nk]*nts, [ntr]*nts, t, Nxik,
                   [k for a in range(nts)])))
        for k in Nk.keys()]
   return max(l, key=lambda x: x[1])[0]

def naiveBayes(train, test):
   '''
   Usage: accuracy = naiveBayes(train, test)
   '''
   test2 = deepcopy(test)
   # number of training examples
   ntr = len(train)
   # number of test examples
   nts = len(test2)
   # transpose training matrix
   m = list(zip(*train))
   # Numerator of P(k)
   Nk = dict(Counter(m[6]))
   # Numerator de P(xi|k)
   Nxik = [NxikCounter(m[i], m[6], Nk.keys()) for i in range(0, len(m)-1)]
   # Classification
   classes = list(map(lambda x: x.pop(6), test2))
   predictions = list(map(classify, test2, [Nk]*nts, [Nxik]*nts, [ntr]*nts))

   return len(list(filter(lambda x: x[0] == x[1],
                          zip(*[predictions, classes])))) / nts * 100
      
   

