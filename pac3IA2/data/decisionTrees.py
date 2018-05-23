from copy import deepcopy
from functools import reduce
from collections import Counter

def bondad(clases, conjunto):
   p = {}
   for i in range(len(clases)):
      p.setdefault(conjunto[i], {})
      p[conjunto[i]].setdefault(clases[i], 0)
      p[conjunto[i]][clases[i]] += 1

   return sum([max([(val, p[atr][val])
                    for val in p[atr].keys()])[1]
               for atr in p.keys()])

def claseMF(clases):
   p = {}
   for x in clases:
      p.setdefault(x, 0)
      p[x] += 1
   return max([(p[cl], cl) for cl in p.keys()])[1]

def iteration2(c):
   c.pop(1)
   return (c[0][0], iteration(list(c[2]), list(zip(*c[1]))))

def iteration(cl, cj):
   l = sorted(set(cl))
   if len(l)==1:
      return ("clase", l[0])
   else:
      (b, col) = max([(bondad(cl, cj[i]), i)
                      for i in range(len(cj))])
      l = cj.pop(col)
      lu = sorted(set(l))
      cj = list(zip(*cj))
      if len(cj) == 0:
         l = [list(filter(lambda x: x[0] == x[1], y))
               for y in [[(val, l[i], cl[i])
                             for i in range(len(l))]
                             for val in lu]]
         return ("atr", col, [(lp[0][0],
                               ("clase", claseMF(list(list(zip(*lp))[2]))))
                              for lp in l])
      elif b == len(cl):
         l = [list(filter(lambda x: x[0] == x[1], y))
               for y in [[(val, l[i], cl[i])
                             for i in range(len(l))]
                             for val in lu]]
         return ("atr", col, [(lp[0][0], ("clase",
                                claseMF(list(list(zip(*lp))[2])))) for lp in l])
      else:   
         l = [list(filter(lambda x: x[0] == x[1], y))
               for y in [[(val, l[i], list(cj[i]), cl[i])
                             for i in range(len(l))]
                             for val in lu]]
         return ('atr', col, [iteration2(list(zip(*x)))
                                  for x in l])

def testEx(ex, tree):
   if isinstance(tree, tuple):
      if tree[0] == "clase":
         return tree[1]
      elif tree[0] == "atr":
         valor = ex.pop(tree[1])
         tree = list(filter(lambda x: x[0][0] == x[1],
                        [(l, valor) for l in tree[2]]))
         if len(tree)==0:
            return None
         else:
            return testEx(ex, tree[0][0][1])
      else:
         return None
   else:
      return None
   


def DTs(train, test):
   atrsTrain = deepcopy(train)
   clasTrain = list(map(lambda x: x.pop(6), atrsTrain))
   atrsTest = deepcopy(test)
   clasTest = list(map(lambda x: x.pop(6), atrsTest))

   tree = iteration(clasTrain, list(zip(*atrsTrain)))
   print(tree)

   preds = [testEx(ex, tree) for ex in atrsTest]

   return len(list(filter(lambda x: x[0] == x[1], zip(*[preds, clasTest])))) / len(clasTest) * 100

