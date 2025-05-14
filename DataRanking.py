# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:16:42 2024

@author: Alexandre José Ferreira

@licence: MIT

    CSV files must contain a header with the names of 
        the criteria and, respectively, in the rows (alternatives), ordinal 
        or cardinal numerical information. If in doubt, access
        the example data.csv

    OBS: Each criterion can be a benefit or a cost. However, to specifically 
        analyze dominance and efficiency, increasing numerical order of the 
        criterion results in greater dominance, that is, the higher the value 
        of the criterion the better (benefit maximization).

requirements : numpy, pandas, sklearn
   
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import evikor

#file=open('data.csv','r')
#file=open('study3.csv','r')
file=open('dataFromKBS.csv','r')

df = pd.read_csv(file)
alt = df['alternatives']
data = df.drop(['alternatives'], axis=1)

pca = PCA(n_components=2)
pca.fit(data.to_numpy())
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

datapca = pd.DataFrame(pca.fit_transform(data), columns=['pc1', 'pc2'])

weights=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]
parameters=[0.25, 0.25, 0.25, 0.25] 
r = evikor.EvikorRanking(data = data, alt = alt, w=weights, p=parameters)
r.__ranking__()

print('rankability: ', r.rho)
print('Data(0_1 reescaled criteria):')
r.dataOutput = r.dataOutput.sort_values('d', ascending=False)
print(r.dataOutput)
print("pairwise matrix(ordering as original data):", r.__DMatrix__()[1])
print(r.__DMatrix__()[0])

dataresult = pd.DataFrame(data, index=r.dataOutput.index)
r1 = evikor.EvikorRanking(data = dataresult, alt = alt, w=weights, p=parameters)
r1.__ranking__()
print("pairwise matrix result (ordering by d):", r1.__DMatrix__()[1])
print(r1.__DMatrix__()[0])


