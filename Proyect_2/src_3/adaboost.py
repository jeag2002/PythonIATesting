from math import log, exp

# parametros
T = 50

# declaracio de funcions

def valClasse(cl):
   if cl == 'B':
      return +1
   else:
      return -1

def hti(atvl, val, igual):
   if (igual and (val == atvl)) or (not igual and (val != atvl)):
      pred = +1
   else:
      pred = -1
   return pred

def error(Dt, cl, atvl, val, igual):
   s = 0
   for i in range(len(cl)):
      if hti(atvl[i], val, igual) != valClasse(cl[i]):
         s += Dt[i]
   return s

def WeakLearner(Dt, tr):
   ll = []
   for atr in range(len(tr)-1):
      for val in sorted(set(tr[atr])):
         et = error(Dt, tr[0], tr[atr], val, True)
         ll.append([et, atr, val, True])
         et = error(Dt, tr[0], tr[atr], val, False)
         ll.append([et, atr, val, False])
   return min(ll)

def testEx(ex, model):
   return sum([regla[0] * hti(ex[regla[2]], regla[3], regla[4])
               for regla in model])

def adaBoost(train, test, verbose=False):
   tt = list(zip(*train))
   m = len(train)
   Dt = [float(1)/float(m)]*m
   model = []
   for i in range(T):
      [et, atr, val, igual] = WeakLearner(Dt, tt)
      if et > 0.5:
         if verbose:
            print("Error: et out of range!")
         break
      if et == 0:
         if verbose:
            print("Zero training error!")
         break
      alfat = (float(1) / float(2)) * log((float(1 - et) / float(et)))
      Dt2 = [Dt[j] * exp(-alfat * hti(tt[atr][j], val, igual) *valClasse(train[j][0]))  for j in range(m)]
      s = sum(Dt2)
      Dt = [(float(Dt2[j])/float(s)) for j in range(m)]
      model.append((alfat, et, atr, val, igual))
   if verbose:
      print(len(model), "regles apreses!")
  
   prediccions = [testEx(ej, model) for ej in test]

   return float(len(list(filter(lambda x: x[0] * valClasse(x[1][0]) > 0, zip(*[prediccions, test]))))) / float(len(test)) 
