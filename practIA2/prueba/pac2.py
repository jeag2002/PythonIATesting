#!/usr/bin/env python
# -*- coding: utf-8 -*-

def pac2(filename):    
    # ------------------------------------------------------------------------------------------------
    # Activity 1: load files
    
    values = list(map(lambda l: [int(x) for x in (l.strip()).split()],
                 (open(filename, 'r').readlines())   ))
    
    # Normalization: substract mean and divide by std deviation
    import numpy
    
    values = numpy.array(values)
    
    prom = list(map(numpy.average, values))
    stds = list(map(numpy.std, values))
    
    values = [list(map(lambda x: (x - prom[i]) / stds[i], values[i]))
             for i in range(len(values))]
    
    
    
    # ------------------------------------------------------------------------------------------------
    # Activity 2
    
    # Apply PCA requesting all components (no argument)
    from sklearn.decomposition import PCA
    mypca = PCA()
    mypca.fit(values)
    
    # How many components are required to explain 95% of the variance
    acumvar = [sum(mypca.explained_variance_ratio_[:i]) for i in range(len(mypca.explained_variance_ratio_))]
    print(list(zip(range(len(acumvar)), acumvar)))
    
    
    # ------------------------------------------------------------------------------------------------
    # Activity 3
    
    # Apply PCA requesting only 2 components
    from sklearn.decomposition import PCA
    mypca = PCA(n_components=2)
    mypca.fit(values)
    values_proj = mypca.transform(values)
    
    # Generate the class tags [0,0,...0, 1, 1, ... 2, ... 9]
    import itertools
    tags = [[i]*200 for i in range(10)]
    tags = list(itertools.chain(*tags))
    
    
    matplotlib.pyplot.scatter(values_proj[:,0],values_proj[:,1],marker='o',c=tags,cmap=cm.jet)
    matplotlib.pyplot.show()
    
    
    # ------------------------------------------------------------------------------------------------
    # Activity 4
    
    from sklearn.lda import LDA
    
    clasif = LDA()
    clasif.fit(values, tags)
    print('LDA clasif score=', clasif.score(values,tags))
    

# Apply the code above to both data files
pac2('mfeat-pix.txt')
pac2('mfeat-fac.txt')
    
    
