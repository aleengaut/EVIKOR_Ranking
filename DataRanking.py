# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:16:42 2024

@author: Alexandre José Ferreira

@licence: MIT

    CSV files must contain a header with the names of 
the criteria and, respectively, in the rows, ordinal 
or cardinal numerical information. If in doubt, access
the example data.csv

requirements : numpy, pandas, sklearn
   
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import rankability
import evikor

#file=open('data_with_alternatives.csv','r')
file=open('study3.csv','r')
df = pd.read_csv(file)
alt = df.pop('alternatives')

'''
PCA must be applied carefully. It should be applied when the \
    rankability of the data is very low and if it is lower than \
        the rankability of the PCs.
'''
#pca = PCA(n_components=2)
#pca.fit(data.to_numpy())
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)

#df = pd.DataFrame(pca.fit_transform(data), columns=['pc1', 'pc2'])

#w=[0.5, 0.5]
#p=[0.25, 0.25, 0.25, 0.25] 
#r = evikor.EvikorRanking(df = df, dfAlt = alt, weights=w, parameters=p)
#r.__evikorRank__()

w=[0.5, 0.5]
p=[0.25, 0.25, 0.25, 0.25]
r = evikor.EvikorRanking(df = df, dfAlt = alt, weights=w, parameters=p)
r.__evikorRankFullInfo__()
print('Rankability index:', r.rankabilityIndex)
print('Data:', r.dfFullInfo)


