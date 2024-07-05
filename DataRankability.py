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

#pca = PCA(n_components=2)
#pca.fit(data.to_numpy())
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)

#df = pd.DataFrame(pca.fit_transform(data), columns=['pc1', 'pc2'])

#dr_1 = rankability.DataRankable(df = df, dfalt = alt)
#dr_1.__rankabilityIndex__()
#dr_1.__sumDomEffVectorAllCriteria__()
#dr_1.__dominanceLevelVector__()
#print('Rankability index:', dr_1.rankabilityIndex)
#print('Data:', dr_1.df)

#w=[0.5, 0.5]
#p=[0.25, 0.25, 0.25, 0.25] 
#r = evikor.EvikorRanking(df = df, dfAlt = alt, weights=w, parameters=p)
#r.__evikorRankFullInfo__()

w=[0.5, 0.5]
p=[0.25, 0.25, 0.25, 0.25] 
#r = evikor.EvikorRanking(df = df, dfAlt = alt, weights=w, parameters=p).__evikorRank__()
r = evikor.EvikorRanking(df = df, dfAlt = alt, weights=w, parameters=p)
r.__evikorRankFullInfo__()
print('Rankability index:', r.rankabilityIndex)
print('Data:', r.dfFullInfo)


