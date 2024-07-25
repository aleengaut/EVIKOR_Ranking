# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:28:23 2024

@author: Alexandre
"""

import numpy as np
from rankability import DataRankable
import math

@staticmethod    
def normalizationDFtoMaxBeZero(df):

    #colDim = self.numberOfAlternatives
    #colDim = df.shape[0]

    for c in df.keys():
        max = df[c].max()
        min = df[c].min()
        delta = max - min
        
        if delta == 0.0: delta = 1.0
        
        df[c] = (max*np.ones(df.shape[0]) -  df[c]) \
                                /delta              
    return df

def domEffRank2Generator(vec):
    
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
        of the criterion the better (benefit maximization).
    '''
    
    def __init__(self, df = None, dfAlt = None, weights = None , parameters = None):
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
     
        super().__init__(df)
        
        try:
            for w in range(weights.__len__()):
                float(weights[w])
            for p in range(parameters.__len__()):
                float(parameters[p])
        except TypeError:
            print('could not convert string to float!')
                
        if parameters[0] + \
                parameters[1] + \
                    parameters[2] + \
                        parameters[3] != 1.0:
            raise Exception('opss..sum constraint: sum of parameters must be 1.')
        
        if np.dot(np.ones(weights.__len__()), weights) != 1.0:
            raise Exception('opss..sum constraint: sum of weights must be 1.')
                
        self.df = df.copy()
        self.dfOriginal = df.copy()
        self.__rankabilityIndex__()
        self.__sumDomEffVectorAllCriteria__()
        self.df[dfAlt.name] = dfAlt
        self.dfAltName = dfAlt.name
        self.weights = weights
        self.parameters = parameters
        
    def __insertDomAndEffRanks__(self):
              
        self.df = self.df.sort_values(['dominance', 'efficiency'], ascending = [False, False])
        i=self.numberOfAlternatives
        rank=[]
        for index in self.df.index:
            rank.append(i)
            i = i - 1
        self.df['domEffRank'] = rank
        self.df['domEffRank'].astype('int64')
        
        dominance = self.df['dominance'].astype('int64')
        countDominance = dominance.value_counts()
                
        self.df['domEffRank2'] = domEffRank2Generator(countDominance)
        self.df['domEffRank2'] = -self.df['domEffRank2'] + \
            self.numberOfAlternatives + 1
        self.df['domEffRank2'].astype('int64')
   
    def __normalizationDataToMax__(self):
        alt = self.df[self.dfAltName]
        self.df = self.df.drop(self.dfAltName, axis=1)
        self.normalized_df = normalizationDFtoMaxBeZero(self.df)
        self.df[alt.name] = alt
        
       # return self.dataNormalized 
        
    def __weigthedSumMinmax__(self):
        
        a = self.df['dominance']
        b = self.df['efficiency']
        t1 = self.df['domEffRank']
        t2 = self.df['domEffRank2']
        alt = self.df[self.dfAltName]
        self.df = self.df.drop(a.name, axis=1)
        self.df = self.df.drop(b.name, axis=1)
        self.df = self.df.drop(t1.name, axis=1)
        self.df = self.df.drop(t2.name, axis=1)
        self.df = self.df.drop(self.dfAltName, axis=1)
        weightSum = np.zeros(self.numberOfAlternatives)
        minmax = np.zeros(self.numberOfAlternatives)
        s = np.zeros(self.numberOfAlternatives)
        j = 0
        for c in self.df.keys():
            sr = s
            s = (self.weights[j]*self.df[c].values)
            minmax = np.maximum(sr, s)
            weightSum = weightSum + s
            j = j + 1
        
        self.df['S'] = weightSum
        self.df['R'] = minmax
        self.df['dominance'] = a
        self.df['efficiency'] = b
        self.df['domEffRank'] = t1
        self.df['domEffRank2'] = t2
        self.df[self.dfAltName] = alt
        
    def __evikorRank__(self):
        self.__insertDomAndEffRanks__()
        self.__normalizationDataToMax__()
        self.__weigthedSumMinmax__()
        
        q = self.parameters[0]*self.df['S'] + \
            self.parameters[1]*self.df['R'] + \
            self.parameters[2]*self.df['domEffRank'] + \
            self.parameters[3]*self.df['domEffRank2']
        
        self.df['q'] = q
        evikorRank = self.df['q'].rank()
        
        out = self.dfOriginal
        out['evikorRank'] = evikorRank.astype('int64')
        out[self.dfAltName] = self.df[self.dfAltName]
        
        self.df = self.df
        self.df = self.dfOriginal
        
        return out
        
    def __evikorRankFullInfo__(self):
        self.__insertDomAndEffRanks__()
        self.__normalizationDataToMax__()
        self.__weigthedSumMinmax__()
        self.__dominanceLevelVector__()
        
        q = self.parameters[0]*self.df['S'] + \
            self.parameters[1]*self.df['R'] + \
            self.parameters[2]*self.df['domEffRank'] + \
            self.parameters[3]*self.df['domEffRank2']
        
        self.df['q'] = q
        evikorRank = self.df['q'].rank()
        
        self.df['evikorRank'] = evikorRank.astype('int64')
        
        self.dfFullInfo = self.df
        self.df = self.dfOriginal
        
