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

weights=[0.5, 0.5]
parameters=[0.25, 0.25, 0.25, 0.25] 
r = evikor.EvikorRanking(data = datapca, alt = alt, w=weights, p=parameters)
r.__ranking__()

print('rankability: ', r.rho)
print('Data(0_1 reescaled criteria):')
print(r.dataOutput)


