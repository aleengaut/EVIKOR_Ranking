# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:28:23 2024

@author: Alexandre
"""

import numpy as np
import pandas as pd
from rankability import DataRankable
import math

@staticmethod    
def normalizationDFtoMaxBeZero(df):

    for c in df.keys():
        max = df[c].max()
        min = df[c].min()
        delta = max - min
        
        if delta == 0.0: delta = 1.0
        
        df[c] = (max*np.ones(df.shape[0]) -  df[c]) \
                                /delta              
    return df

def normalizationDFtoMin0Max1(df):
    
    for c in df.keys():
        max = df[c].max()
        min = df[c].min()
        delta = max - min
        
        if delta == 0.0: delta = 1.0

        df[c] = (df[c] - min*np.ones(df.shape[0])) \
                                /delta              
    return df

def t2Generator(vec):
    
    odd = 0
    rank = []
    i=1
    p=1
    for c in vec:
    	if c % 2 == 0: 
    		odd = 1
    		freq = int(c/2)
    	else:
    		odd = 0
    		freq = math.ceil(c/2 - 1)
    		
    	if odd == 1:
    		for f in range(freq):
    			if p == 1:
    				rank.append(i+1)
    				p = p + 1
    			if p == 2:
    				rank.append(i)
    				p = 1
    				
    			i = i + 2
    			
    	if odd == 0 and freq > 0:
    		for f in range(freq):
    			if p == 1:
    				rank.append(i+1)
    				p = p + 1
    			if p == 2:
    				rank.append(i)
    				p = 1
    
    			i = i + 2
    			rank.append(i)
    			
    	if odd == 0 and freq == 0:
    		rank.append(i) 
    		i = i + 1
                
    return rank

class EvikorRanking(DataRankable):
    '''Class for estimating data rankability using the rho method 
    proposed by Alexandre and Amit.
    
    requirements : numpy
    input: alternative-criteria dataframe AND alternative column name
        
    OBS: Each criterion can be a benefit or a cost. However, to specifically 
        analyze dominance and efficiency, increasing numerical order of the 
        criterion results in greater dominance, that is, the higher the value 
        of the criterion the better.
    '''
    
    def __init__(self, data = None, alt = None, w = None , p = None):
        '''
        Parameters
        ----------
        df : TYPE, optional
            DESCRIPTION. The default is None.
        alt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        
        # Normalizad criteria max value -> 1.
        data=normalizationDFtoMin0Max1(data)
        
        super().__init__(data, alt)
        
        try:
            for wj in range(w.__len__()):
                float(w[wj])
            for pi in range(p.__len__()):
                float(p[pi])
        except TypeError:
            print('could not convert string to float!')
                
        p=np.array(p)
        p=p/p.sum()
        self.p = p
        
        w=np.array(w)
        w=w/w.sum()
        self.w = w
        
        self.__rankabilityIndex__()
        self.__sumDominanceVectorAllCriteria__()
        self.__sumDomEffVectorAllCriteria__()
        
    def __insertRankst__(self):
              
        self.dataOutput = self.dataOutput.sort_values(['d', 'lambda'], ascending = [True, True])
        i=self.numberOfAlternatives
        rank=[]
        for index in self.dataOutput.index:
            rank.append(i)
            i = i - 1
        self.dataOutput['t'] = rank
        self.dataOutput['t'].astype('int64')
        
        dominance = self.dataOutput['d'].astype('int64')
        countDominance = dominance.value_counts()
                
        self.dataOutput['t2'] = t2Generator(countDominance)
        self.dataOutput['t2'] = -self.dataOutput['t2'] + \
            self.numberOfAlternatives + 1
        self.dataOutput['t2'].astype('int64')
   
    def __weigthedSumMinmax__(self):
        
        weightSum = np.zeros(self.numberOfAlternatives)
        minmax = np.zeros(self.numberOfAlternatives)
        s = np.zeros(self.numberOfAlternatives)
        
        # Normalizad criteria max value -> 0.
        self.data = normalizationDFtoMaxBeZero(self.data)
        j = 0
        for c in self.data.keys():
            sr = s
            s = (self.w[j]*self.data[c].values)
            minmax = np.maximum(sr, s)
            weightSum = weightSum + s
            j = j + 1
        
        self.dataOutput['S'] = weightSum
        self.dataOutput['R'] = minmax

        
    def __ranking__(self):
        self.__weigthedSumMinmax__()
        self.__insertRankst__()
        
        df = pd.DataFrame(data=self.dataOutput['S'])
        df['R'] = self.dataOutput['R']
        df['t'] = self.dataOutput['t']
        df['t2'] = self.dataOutput['t2']
        df = normalizationDFtoMin0Max1(df)
        
        k = 0
        q = np.zeros(self.numberOfAlternatives)
        for r in df.keys():
            q = q + self.p[k]*df[r]
            k = k + 1 
        
        self.dataOutput['q'] = q
        self.q = q
        evikorRank = self.dataOutput['q'].rank()
        
        self.dataOutput['Ranking'] = evikorRank.astype('int64')
        self.ranking = evikorRank
        
        self.dataOutput['S'] = df['S']
        self.dataOutput['R'] = df['R']